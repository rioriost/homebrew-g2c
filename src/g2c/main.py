#!/usr/bin/env python3
import argparse
import os
import sys
import json
import re
import tokenize
import ast
import urllib.request
import io
import importlib.resources
import logging
from typing import List, Tuple

from antlr4 import InputStream, CommonTokenStream
from .CypherLexer import CypherLexer
from .CypherParser import CypherParser
from .CypherVisitor import CypherVisitor

# Import the OpenAI client library after installing openai.
from openai import OpenAI

# Configure logging (only errors will be printed)
logging.basicConfig(level=logging.ERROR)


class QueryExtractor:
    """
    Extracts Gremlin queries along with their line numbers from
    source code files. It supports Python source files (using AST
    or tokenize methods) and generic source files (Java, C#, etc.)
    using regular expressions.
    """

    def __init__(self, code: str, file_path: str):
        """
        Initialize the extractor with source code and its file path/URL.
        """
        self.code: str = code
        self.file_path: str = file_path

    def extract(self) -> List[Tuple[int, str]]:
        """
        Determine file type and choose the corresponding extraction method.
        Returns a list of tuples (line_number, query_literal).
        """
        # For Python files, decide which extraction method to use.
        if self.file_path.endswith(".py"):
            if (
                "from gremlin_python" in self.code
                or "import gremlin_python" in self.code
            ):
                return self._extract_gremlin_queries()
            else:
                return self._extract_gremlin_from_literals()
        else:
            return self._extract_generic_literals()

    def _extract_gremlin_from_literals(self) -> List[Tuple[int, str]]:
        """
        For Python files that do not import gremlin_python,
        tokenizes the source and looks for string literals starting with "g.".
        Returns a list of (line_number, literal) tuples.
        """
        results: List[Tuple[int, str]] = []
        try:
            # Convert the code string to bytes for tokenize.
            readline = io.BytesIO(self.code.encode("utf-8")).readline
            tokens = tokenize.tokenize(readline)
            for token in tokens:
                if token.type == tokenize.STRING:
                    literal = token.string
                    # Check if the literal (or with opening quote removed) starts with "g."
                    if literal.startswith("g.") or (
                        len(literal) > 1 and literal[1:].startswith("g.")
                    ):
                        # token.start returns (line, column); we want the first element.
                        results.append((token.start[0], literal.strip('"')))
        except Exception as e:
            sys.exit("Error while parsing Python file: " + str(e))
        return results

    def _extract_gremlin_queries(self) -> List[Tuple[int, str]]:
        """
        Uses an AST visitor to locate call nodes that originate from identifier 'g'.
        Returns a list of (line_number, query_string) tuples.
        """
        try:
            tree = ast.parse(self.code)
        except Exception as e:
            sys.exit("Error while parsing Python file: " + str(e))

        class GremlinQueryVisitor(ast.NodeVisitor):
            def __init__(self):
                self.queries: List[Tuple[int, str]] = []

            def visit_Call(self, node):
                if self._starts_with_g(node.func):
                    try:
                        # Try to convert the AST node back into actual source code.
                        query_str = ast.unparse(node)
                    except Exception:
                        query_str = self._node_to_str(node)
                    lineno = getattr(node, "lineno", -1)
                    self.queries.append((lineno, query_str))
                self.generic_visit(node)

            def _starts_with_g(self, node):
                """Check recursively if the AST node starts with 'g'."""
                if isinstance(node, ast.Name):
                    return node.id == "g"
                elif isinstance(node, ast.Attribute):
                    return self._starts_with_g(node.value)
                elif isinstance(node, ast.Call):
                    return self._starts_with_g(node.func)
                return False

            def _node_to_str(self, node):
                """Fallback conversion for an AST node."""
                return "<Gremlin query representation>"

        visitor = GremlinQueryVisitor()
        visitor.visit(tree)
        return visitor.queries

    def _extract_generic_literals(self) -> List[Tuple[int, str]]:
        """
        Removes comment blocks from non-Python source code files and
        uses regular expressions to find double-quoted literals starting with "g.".
        Returns a list of (line_number, literal) tuples.
        """
        # Remove single-line and multi-line comments.
        code_no_comments = re.sub(r"//.*", "", self.code)
        code_no_comments = re.sub(r"/\*.*?\*/", "", code_no_comments, flags=re.DOTALL)

        results: List[Tuple[int, str]] = []
        # RegEx to match a double-quoted string starting with "g." (handles escaped quotes)
        string_pattern = r'"g\.(?:\\.|[^"\\])*"'
        for match in re.finditer(string_pattern, code_no_comments):
            literal = match.group(0)
            # Calculate the line number based on newline count before the match.
            lineno = code_no_comments.count("\n", 0, match.start()) + 1
            results.append((lineno, literal))
        return results


class CacheManager:
    """
    Manages caching of Gremlin-to-Cypher conversion results.
    Cache is stored in a JSON file at ~/.g2c_cache.
    """

    def __init__(self):
        self.cache_file = os.path.join(os.path.expanduser("~"), ".g2c_cache")
        self.cache = self._load_cache()

    def _load_cache(self) -> dict:
        """
        Loads the cache from disk.
        If the cache file does not exist, deploy a default one.
        """
        if not os.path.exists(self.cache_file):
            self.deploy_g2c_cache()
        try:
            with open(self.cache_file, "r", encoding="utf-8") as file:
                return json.load(file)
        except Exception as e:
            sys.exit("Error loading cache file: " + str(e))

    def _save_cache(self) -> None:
        """
        Saves the cache dictionary to disk.
        """
        try:
            with open(self.cache_file, "w", encoding="utf-8") as file:
                json.dump(self.cache, file, ensure_ascii=False, indent=4)
        except Exception as e:
            sys.exit("Error saving cache file: " + str(e))

    def add_search_result(self, keyword: str, result: str) -> None:
        """
        Add a new conversion result to the cache.
        """
        self.cache[keyword] = result
        self._save_cache()

    def get_search_result(self, keyword: str) -> str:
        """
        Retrieve a conversion result from the cache.
        Returns None if not found.
        """
        return self.cache.get(keyword, None)

    def deploy_g2c_cache(self) -> None:
        """
        Deploy the initial .g2c_cache file from packaged data if it exists.
        """
        target_path = self.cache_file
        try:
            # Open the packaged cache file using importlib.resources.
            with importlib.resources.open_binary(
                "g2c.packaged_data", ".g2c_cache"
            ) as resource_file:
                cache_data = resource_file.read()
            with open(target_path, "wb") as f:
                f.write(cache_data)
            print(f".g2c_cache was deployed to {target_path}.")
        except FileNotFoundError:
            print(".g2c_cache not found in packaged data.")
        except Exception as e:
            print(f"Error deploying cache: {e}")


class GremlinToCypherConverter:
    """
    Uses the OpenAI API to convert a Gremlin query into its equivalent Cypher query.
    """

    KEY_ENV_NAME = "G2C_OPENAI_API_KEY"

    def __init__(self):
        """
        Initialize the converter by retrieving the API key from the environment.
        """
        try:
            self.api_key = os.environ[self.KEY_ENV_NAME]
        except KeyError:
            sys.exit(
                f"API key not found. Please set the {self.KEY_ENV_NAME} environment variable."
            )
        # Initialize the OpenAI client with the API key.
        self.client = OpenAI(api_key=self.api_key)

    def convert(self, gremlin_query: str) -> str:
        """
        Convert the supplied Gremlin query to a Cypher query.
        Returns the Cypher query as a string.
        """
        # Create a prompt that includes the Gremlin query.
        prompt = f"""Convert the following Gremlin query to an equivalent Cypher query:

{gremlin_query}
"""
        try:
            # Call the OpenAI Chat Completion API.
            chat_completion = self.client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a helpful assistant that converts Gremlin queries "
                            "into Cypher queries. You reply with only the converted Cypher query."
                            "Do not include any additional text or explanations."
                            "Do not include carriage returns or line breaks."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                model="gpt-4o-mini",
                temperature=0.0,  # Deterministic conversions.
            )

            logging.debug(f"API response: {chat_completion}")

            # Extract and return the Cypher query.
            cypher_query = chat_completion.choices[0].message.content
            if cypher_query:
                return cypher_query
            else:
                raise ValueError("No Cypher query found in the API response.")
        except Exception as e:
            print(f"Error during conversion: {e}")
            return ""


class GremlinConverterController:
    """
    Controller class to coordinate reading input, extracting Gremlin queries,
    converting them to Cypher, and outputting the results.
    """

    def __init__(self, args: argparse.Namespace):
        """
        Initialize with parsed command line arguments.
        """
        self.args = args
        self.cache_manager = CacheManager()
        self.converter = GremlinToCypherConverter()

    def _read_code(self) -> Tuple[str, str]:
        """
        Reads code from a file, URL, or directly from the -g/--gremlin argument.
        Returns a tuple (code, path), where path is the file or URL or "direct query".
        """
        code = ""
        path = ""
        if self.args.filepath:
            # Read code from the provided file path.
            if os.path.exists(self.args.filepath):
                path = self.args.filepath
                try:
                    with open(self.args.filepath, "r", encoding="utf-8") as f:
                        code = f.read()
                except Exception as e:
                    sys.exit("Failed to read the file: " + str(e))
            else:
                sys.exit(f"File not found: {self.args.filepath}")
        elif self.args.url:
            # Read code from the provided URL.
            try:
                with urllib.request.urlopen(self.args.url) as response:
                    code = response.read().decode("utf-8")
                path = self.args.url
            except Exception as e:
                sys.exit("Failed to fetch the file: " + str(e))
        elif self.args.gremlin:
            # Direct conversion of the single Gremlin query.
            path = "direct query"
            # Replace smart quotes with standard double quotes.
            code = self.args.gremlin.replace("“", '"').replace("”", '"')
        else:
            sys.exit("No valid input provided.")
        return code.strip('"'), path

    def process(self) -> None:
        """
        Main processing method:
          1. Reads the code or query.
          2. Extracts Gremlin queries.
          3. For each query, check cache and then convert if needed.
          4. Prints the converted Cypher queries.
        """
        code, path = self._read_code()

        extracted: List[Tuple[int, str]] = []
        # If input is a file or URL, extract potential queries
        if path != "direct query":
            extractor = QueryExtractor(code, path)
            extracted = extractor.extract()
            if not extracted:
                sys.exit(f"No Gremlin queries found in the source code: {path}")
        else:
            # Wrap the supplied query in a list with an artificial line number.
            extracted = [(1, code)]

        cypher_queries: dict = {}
        # Process each extracted query.
        for lineno, query in extracted:
            # First try retrieving a cached conversion.
            cypher_result = self.cache_manager.get_search_result(query)
            if cypher_result:
                cypher_queries[f"line {lineno}, {query}"] = self.format_cypher(
                    cypher_result
                )
            else:
                # Convert the query using OpenAI API.
                cypher_query = self.converter.convert(query)
                if cypher_query:
                    # Add to cache.
                    self.cache_manager.add_search_result(query, cypher_query)
                    cypher_queries[f"line {lineno}, {query}"] = self.format_cypher(
                        cypher_query
                    )
                else:
                    cypher_queries[f"line {lineno}, {query}"] = self.format_cypher("")

        # Print all conversion results.
        print("Converted Cypher queries:\n")
        for src, cypher_query in cypher_queries.items():
            print(f"{src} ->\n{cypher_query}\n")

    def format_cypher(self, cypher_query: str) -> str:
        """
        Format a Cypher query
        If self.args.age is True, format the query for Apache AGE.
        """
        # Color definitions for pretty-printing output.
        RED = "\033[0;31m"
        GREEN = "\033[0;32m"
        RESET = "\033[0m"
        # Failed to conver
        if not cypher_query:
            return f"{RED}[Failed]{RESET}"

        # for Apache AGE
        if self.args.age:
            cypher_for_age = self.format_for_age(cypher_query)
            return f"{GREEN}{cypher_for_age}{RESET}"

        # standard cypher query
        return f"{GREEN}{cypher_query}{RESET}"

    @staticmethod
    def format_for_age(cypher_query: str) -> str:
        logging.debug(cypher_query)
        lexer = CypherLexer(InputStream(cypher_query))
        parser = CypherParser(CommonTokenStream(lexer))
        tree = parser.oC_Cypher()
        visitor = MyCypherVisitor()
        result = visitor.visit(tree)
        if result is None:
            ag_types = ", ".join([f"{p} agtype" for p in visitor.expressions])
            return f"SELECT * FROM cypher('GRAPH_NAME', $$ {cypher_query} $$) AS ({ag_types});"
        else:
            return cypher_query


class MyCypherVisitor(CypherVisitor):
    def __init__(self):
        self.expressions: list = []
        self.aliases: dict = {}
        self.re = re.compile(
            r"^[a-zA-Z][a-zA-Z0-9\-\._]*$"
        )  # I'm not sure the naming rule is correct or not

    def visitOC_Cypher(self, ctx: CypherParser.OC_CypherContext):
        # root node
        logging.debug("visitOC_Cypher")
        # visit children
        return self.visitChildren(ctx)

    def visitOC_Statement(self, ctx: CypherParser.OC_StatementContext):
        logging.debug("visitOC_Statement")
        return self.visitChildren(ctx)

    def visitOC_Query(self, ctx: CypherParser.OC_QueryContext):
        logging.debug("visitOC_Query")
        return self.visitChildren(ctx)

    def visitOC_RegularQuery(self, ctx: CypherParser.OC_RegularQueryContext):
        logging.debug("visitOC_RegularQuery")
        return self.visitChildren(ctx)

    def visitOC_Union(self, ctx: CypherParser.OC_UnionContext):
        logging.debug("visitOC_Union")
        return self.visitChildren(ctx)

    def visitOC_SingleQuery(self, ctx: CypherParser.OC_SingleQueryContext):
        logging.debug("visitOC_SingleQuery")
        return self.visitChildren(ctx)

    def visitOC_SinglePartQuery(self, ctx: CypherParser.OC_SinglePartQueryContext):
        logging.debug("visitOC_SinglePartQuery")
        return self.visitChildren(ctx)

    def visitOC_Return(self, ctx: CypherParser.OC_ReturnContext):
        # Here is the visitOC_Return method. Available to add ProjectionBody and other methods
        logging.debug("visitOC_Return")
        return self.visitChildren(ctx)

    def visitOC_ProjectionBody(self, ctx: CypherParser.OC_ProjectionBodyContext):
        logging.debug("visitOC_ProjectionBody")
        return self.visitChildren(ctx)

    def visitOC_ProjectionItems(self, ctx: CypherParser.OC_ProjectionItemsContext):
        logging.debug("visitOC_ProjectionItems")
        return self.visitChildren(ctx)

    def visitOC_ProjectionItem(self, ctx: CypherParser.OC_ProjectionItemContext):
        # with this rule, we can extract the expression and alias separately because it follows the pattern "expression [AS variable]".
        exprText = ctx.getChild(0).getText()
        aliasText = None
        if ctx.getChildCount() > 2:
            # Generally, the third child is the variable name (after the AS clause)
            aliasText = ctx.getChild(4).getText()
            logging.debug(
                "ProjectionItem: expression =", exprText, ", alias =", aliasText
            )
            self.aliases[exprText] = aliasText
        else:
            logging.debug("ProjectionItem: expression =", exprText)
        return self.visitChildren(ctx)

    def visitOC_Match(self, ctx: CypherParser.OC_MatchContext):
        logging.debug("visitOC_Match")
        return self.visitChildren(ctx)

    def visitOC_Expression(self, ctx: CypherParser.OC_ExpressionContext):
        exp = ctx.getText()
        logging.debug("visitOC_Expression: ", exp)
        if exp in self.aliases.keys():
            exp = self.aliases[exp]
        if "." in exp:
            exp = exp.split(".")[1]
        if exp and exp not in self.expressions:
            if re.match(self.re, exp):
                self.expressions.append(exp)
        return self.visitChildren(ctx)

    def visitOC_Literal(self, ctx: CypherParser.OC_LiteralContext):
        literal = ctx.getText()
        logging.debug("visitOC_Literal: ", literal)
        return self.visitChildren(ctx)


def parse_arguments() -> argparse.Namespace:
    """
    Parses command line arguments.
    The user must provide either: a Gremlin query, a file path, or a URL.
    """
    parser = argparse.ArgumentParser(
        description="Convert Gremlin queries to Cypher queries."
    )
    parser.add_argument(
        "-a",
        "--age",
        action="store_true",
        default=False,
        help="Convert to the Cypher query for Apache AGE.",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "-g", "--gremlin", type=str, help="The Gremlin query to convert."
    )
    group.add_argument(
        "-f", "--filepath", help="Path to the source code file (.py, .java, .cs, .txt)"
    )
    group.add_argument(
        "-u", "--url", help="URL to the source code file (.py, .java, .cs, .txt)"
    )
    return parser.parse_args()


def main() -> None:
    """
    Entry point of the script.
    Parses arguments, initializes the controller, and starts processing.
    """
    args = parse_arguments()
    controller = GremlinConverterController(args)
    controller.process()


if __name__ == "__main__":
    main()
