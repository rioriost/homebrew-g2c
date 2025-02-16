#!/usr/bin/env python3
import argparse
import os
import sys
from openai import OpenAI
import logging
import json
import re
import tokenize
import ast
import urllib.request
from typing import List, Tuple
import io
import importlib.resources

logging.basicConfig(level=logging.ERROR)


class QueryExtractor:
    """
    A class to extract Gremlin queries and their line numbers from source code files.
    """

    def __init__(self, code: str, file_path: str):
        """
        Initialize an instance with the path to the source code file.

        Arguments:
            code: The source code as a string.
            file_path: Path or URL to the file that will be analyzed.
        """
        self.code: str = code
        self.file_path: str = file_path

    def extract(self) -> List[Tuple[str, int]]:
        """
        Determine the file type based on its extension and extract Gremlin queries accordingly.

        For Python files:
            • If the file imports 'gremlin_python', use the AST based method.
            • Otherwise, use an alternate method which looks for string literals starting with "g.".
        For Other files:
            • Use a generic extraction method based on regular expressions after removing comments.

        Returns:
            A list of (literal, line_number) tuples.
        """
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

    def _extract_gremlin_from_literals(self) -> List[Tuple[str, int]]:
        """
        Extract Gremlin queries from Python source files that do not explicitly import gremlin_python.
        Scans using tokenize and records string literals beginning with "g." along with their line numbers.
        Returns:
          A list of (literal, line_number) tuples.
        """
        results: List[Tuple[str, int]] = []
        try:
            # tokenize.tokenize() requires a bytes iterator.
            # Wrap the code string in a BytesIO object.
            readline = io.BytesIO(self.code.encode("utf-8")).readline
            tokens = tokenize.tokenize(readline)
            for token in tokens:
                # Process only string tokens.
                if token.type == tokenize.STRING:
                    # Remove any quotes from the token string if necessary. You may want to use ast.literal_eval(token.string)
                    # to get the actual string content.
                    literal = token.string
                    if literal.startswith("g.") or (
                        literal[1:].startswith("g.") if len(literal) > 1 else False
                    ):
                        # token.start is a tuple: (start_column, start_line)
                        results.append((token.start[0], literal))
        except Exception as e:
            sys.exit("Error while parsing Python file: " + str(e))
        return results

    def _extract_gremlin_queries(self) -> List[Tuple[str, int]]:
        """
        Extract Gremlin queries from Python source files that use gremlin_python objects.
        Uses an AST visitor to locate call chains that begin from the identifier "g".
        Returns:
          A list of (query_string, line_number) tuples.
        """
        results: List[Tuple[int, str]] = []
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
                        # Convert the AST node back into source code if possible
                        query_str = ast.unparse(node)
                    except Exception:
                        query_str = self._node_to_str(node)
                    lineno = getattr(node, "lineno", -1)
                    self.queries.append((lineno, query_str))
                self.generic_visit(node)

            def _starts_with_g(self, node):
                """
                Recursively check if the given AST node represents an expression
                that begins with the identifier 'g'.
                """
                if isinstance(node, ast.Name):
                    return node.id == "g"
                elif isinstance(node, ast.Attribute):
                    return self._starts_with_g(node.value)
                elif isinstance(node, ast.Call):
                    return self._starts_with_g(node.func)
                return False

            def _node_to_str(self, node):
                """
                Fallback conversion of an AST node to a string representation.
                """
                return "<Gremlin query representation>"

        visitor = GremlinQueryVisitor()
        visitor.visit(tree)
        results = visitor.queries
        return results

    def _extract_generic_literals(self) -> List[Tuple[str, int]]:
        """
        Extract Gremlin query string literals from Java or C# source files.
        First removes comment blocks, then finds double-quoted string literals that start with "g.".
        It also computes the line number of each occurrence.
        Returns:
          A list of (literal, line_number) tuples.
        """
        # Remove single-line (//...) and multi-line (/* ... */) comments.
        code_no_comments = re.sub(r"//.*", "", self.code)
        code_no_comments = re.sub(r"/\*.*?\*/", "", code_no_comments, flags=re.DOTALL)

        results: List[Tuple[str, int]] = []

        # Regular expression for matching a double-quoted string starting with "g."
        # It accounts for escaped quotes.
        string_pattern = r'"g\.(?:\\.|[^"\\])*"'
        for match in re.finditer(string_pattern, code_no_comments):
            literal = match.group(0)
            # Compute line number by counting newlines from beginning to match's start.
            lineno = code_no_comments.count("\n", 0, match.start()) + 1
            results.append((lineno, literal))
        return results


class CacheManager:
    def __init__(self):
        self.cache_file = os.path.join(os.path.expanduser("~"), ".g2c_cache")
        self.cache = self._load_cache()

    def _load_cache(self):
        if not os.path.exists(self.cache_file):
            self.deploy_g2c_cache()
        with open(self.cache_file, "r") as file:
            return json.load(file)

    def _save_cache(self):
        with open(self.cache_file, "w") as file:
            json.dump(self.cache, file, ensure_ascii=False, indent=4)

    def add_search_result(self, keyword: str, result: str):
        self.cache[keyword] = result
        self._save_cache()

    def get_search_result(self, keyword: str) -> str:
        return self.cache.get(keyword, None)

    def deploy_g2c_cache(self):
        target_path = self.cache_file
        try:
            # read .g2c_cache in packaged_data directory
            with importlib.resources.open_binary(
                "g2c.packaged_data", ".g2c_cache"
            ) as resource_file:
                cache_data = resource_file.read()

            with open(target_path, "wb") as f:
                f.write(cache_data)
            print(f".g2c_cache was deployed to {target_path}.")
        except FileNotFoundError:
            print(".g2c_cache not found.")
        except Exception as e:
            print(f"Error: {e}")


class GremlinToCypherConverter:
    """
    A class to convert Gremlin queries into Cypher queries using the OpenAI API.
    """

    KEY_ENV_NAME = "G2C_OPENAI_API_KEY"

    def __init__(self):
        # Retrieve the API key from the environment.
        try:
            self.api_key = os.environ[self.KEY_ENV_NAME]
        except KeyError:
            sys.exit(
                f"API key not found. Please set the {self.KEY_ENV_NAME} environment variable."
            )

        # Initialize the OpenAI client with the obtained API key.
        self.client = OpenAI(api_key=self.api_key)

    def convert(self, gremlin_query: str) -> str:
        """
        Converts a Gremlin query into a Cypher query using the OpenAI API.

        Parameters:
            gremlin_query (str): The Gremlin query to convert.

        Returns:
            str: The converted Cypher query.
        """
        # Create the prompt for conversion including the Gremlin query.
        prompt = f"""
        Convert the following Gremlin query to an equivalent Cypher query:

        {gremlin_query}
        """

        try:
            # Call the Azure OpenAI Chat Completion API with a system message and the user prompt.
            chat_completion = self.client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a helpful assistant that converts Gremlin queries "
                            "into Cypher queries. You reply just a converted Cypher query without quotes."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                model="gpt-4o-mini",
                temperature=0.0,  # Setting temperature to 0 for deterministic output.
            )

            logging.debug(f"API response: {chat_completion}")

            # Extract and return the Cypher query from the API response.
            cypher_query = chat_completion.choices[0].message.content
            if cypher_query:
                return cypher_query
            else:
                raise ValueError("No Cypher query found in the API response.")
        except Exception as e:
            print(f"Error during conversion: {e}")
            return ""


def main() -> None:
    RED = "\033[0;31m"
    GREEN = "\033[0;32m"
    RESET = "\033[0m"

    parser = argparse.ArgumentParser(
        description="Convert Gremlin queries to Cypher queries."
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
    args = parser.parse_args()

    code = ""

    if args.filepath:
        if os.path.exists(args.filepath):
            path = args.filepath
            try:
                with open(args.filepath, "r", encoding="utf-8") as f:
                    code = f.read()
            except Exception as e:
                sys.exit("Failed to read the file: " + str(e))
        else:
            print(f"File not found: {args.filepath}")
            return

    if args.url:
        try:
            with urllib.request.urlopen(args.url) as response:
                code = response.read().decode("utf-8")
            path = args.url
        except Exception as e:
            sys.exit("Failed to fetch the file: " + str(e))

    if code:
        extractor = QueryExtractor(code, path)
        extracted = extractor.extract()
        if not extracted:
            print(f"No Gremlin queries found in the source code: {path}.")
            return

    if args.gremlin:
        extracted = [(1, args.gremlin.replace("“", '"').replace("”", '"'))]

    cypher_queries: dict = {}
    for lineno, query in extracted:
        cm = CacheManager()
        cypher_result = cm.get_search_result(query)
        if cypher_result:
            cypher_queries[f"line {lineno}, {query}"] = f"{GREEN}{cypher_result}{RESET}"
        else:
            converter = GremlinToCypherConverter()
            cypher_query = converter.convert(query)
            if cypher_query:
                cm.add_search_result(query, cypher_query)
                cypher_queries[f"line {lineno}, {query}"] = (
                    f"{GREEN}{cypher_query}{RESET}"
                )
            else:
                cypher_queries[f"line {lineno}, {query}"] = f"{RED}[Failed]{RESET}"

    if cypher_queries:
        print("Converted Cypher queries:\n")
        for src, cypher_query in cypher_queries.items():
            print(f"{src} ->\n{cypher_query}\n")


if __name__ == "__main__":
    main()
