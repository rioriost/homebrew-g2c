#!/usr/bin/env python3
import argparse
import json
import os
import sys
import unittest
from unittest.mock import patch, mock_open, MagicMock

# Import the module under test.
# (Adjust the import if your package layout is different.)
from g2c.main import (
    QueryExtractor,
    CacheManager,
    GremlinToCypherConverter,
    GremlinConverterController,
    MyCypherVisitor,
    parse_arguments,
    main,
)


# === Test QueryExtractor ===
class TestQueryExtractor(unittest.TestCase):
    def test_extract_gremlin_from_literals(self):
        # Code that does not import gremlin_python so that _extract_gremlin_from_literals is used.
        # Prepare a simple python code with a string literal starting with "g.".
        code = 'print("g.foo")'
        qe = QueryExtractor(code, "dummy.py")
        # The tokenize method will produce a token whose string is "g.foo"
        result = qe.extract()
        # Expect that the extracted tuple contains the literal without quotes.
        self.assertEqual(len(result), 1)
        line, query = result[0]
        self.assertIsInstance(line, int)
        self.assertEqual(query, "g.foo")

    def test_extract_gremlin_queries_with_gremlin_import(self):
        # When the source contains "from gremlin_python", extraction should use the AST visitor.
        # We craft a code sample that has an import and a function call starting with g.
        code = "from gremlin_python import something\ng.query_method(arg=1)"
        qe = QueryExtractor(code, "dummy.py")
        result = qe.extract()
        # The visitor will add the call node.
        # We expect at least one extracted query whose string representation
        # includes “g.query_method(” (the exact format depends on ast.unparse).
        self.assertTrue(any("g.query_method(" in q for _, q in result))

    def test_extract_generic_literals(self):
        # For non-python files, extraction uses the generic regexp.
        code = 'Some code "g.bar(1)" more code'
        qe = QueryExtractor(code, "dummy.java")
        result = qe.extract()
        self.assertEqual(len(result), 1)
        line, literal = result[0]
        self.assertEqual(line, 1)
        # Literal remains with enclosing double quotes.
        self.assertEqual(literal, '"g.bar(1)"')


# === Test CacheManager ===
class TestCacheManager(unittest.TestCase):
    @patch("os.path.exists", return_value=True)
    @patch(
        "builtins.open",
        new_callable=mock_open,
        read_data='{"test_query": "test_result"}',
    )
    def test_load_existing_cache(self, m_open, m_exists):
        cm = CacheManager()
        self.assertEqual(cm.cache, {"test_query": "test_result"})

    @patch("os.path.exists", return_value=False)
    @patch("g2c.main.CacheManager.deploy_g2c_cache")
    @patch("builtins.open", new_callable=mock_open, read_data="{}")
    def test_load_cache_triggers_deploy(self, m_open, m_deploy, m_exists):
        # When cache file does not exist, deploy_g2c_cache is called.
        cm = CacheManager()
        m_deploy.assert_called_once()
        self.assertEqual(cm.cache, {})

    @patch("builtins.open", new_callable=mock_open, read_data="{}")
    def test_save_cache_and_add_result(self, m_open):
        cm = CacheManager()
        # Set cache manually then add result.
        cm.cache = {}
        cm.add_search_result("new_query", "new_result")
        self.assertEqual(cm.cache, {"new_query": "new_result"})
        # Make sure file write was called. (The call_args_list concatenated should match JSON dump.)
        handle = m_open()
        written_calls = handle.write.call_args_list
        written_data = "".join(call.args[0] for call in written_calls)
        expected_data = json.dumps(
            {"new_query": "new_result"}, ensure_ascii=False, indent=4
        )
        self.assertEqual(written_data, expected_data)

    @patch(
        "builtins.open",
        new_callable=mock_open,
        read_data='{"test_query": "test_result"}',
    )
    def test_get_search_result(self, m_open):
        cm = CacheManager()
        self.assertEqual(cm.get_search_result("test_query"), "test_result")
        self.assertIsNone(cm.get_search_result("nonexistent"))

    @patch("importlib.resources.open_binary", side_effect=FileNotFoundError())
    @patch("os.path.exists", return_value=False)
    @patch("builtins.print")
    def test_deploy_g2c_cache_file_not_found(self, m_print, m_exists, m_open_binary):
        # Call deploy_g2c_cache and see that FileNotFoundError is handled.
        cm = CacheManager()
        # When deploy_g2c_cache is called and file not found, it should print a message.
        # Because __init__ calls deploy_g2c_cache if file missing, m_print should be called.
        m_print.assert_any_call(".g2c_cache not found in packaged data.")


# === Test GremlinToCypherConverter ===
class TestGremlinToCypherConverter(unittest.TestCase):
    @patch.dict(os.environ, {"G2C_OPENAI_API_KEY": "fake_api_key"})
    @patch("g2c.main.OpenAI")
    def test_convert_success(self, mock_openai):
        # Fake the openai client and response.
        fake_client = MagicMock()
        mock_openai.return_value = fake_client
        fake_response = MagicMock()
        fake_choice = MagicMock()
        fake_choice.message.content = "MATCH (n) RETURN n"
        fake_response.choices = [fake_choice]
        fake_client.chat.completions.create.return_value = fake_response

        converter = GremlinToCypherConverter()
        result = converter.convert("g.V()")
        self.assertEqual(result, "MATCH (n) RETURN n")

    @patch.dict(os.environ, {}, clear=True)
    def test_missing_api_key(self):
        # When the environment variable is missing, the converter should call sys.exit.
        with self.assertRaises(SystemExit) as cm:
            GremlinToCypherConverter()
        self.assertIn("API key not found.", str(cm.exception))

    @patch.dict(os.environ, {"G2C_OPENAI_API_KEY": "fake_api_key"})
    @patch("g2c.main.OpenAI")
    def test_convert_exception(self, mock_openai):
        fake_client = MagicMock()
        mock_openai.return_value = fake_client
        fake_client.chat.completions.create.side_effect = Exception("API error")
        converter = GremlinToCypherConverter()
        # When an exception occurs the method prints error and returns empty string.
        result = converter.convert("g.V()")
        self.assertEqual(result, "")


# === Test GremlinConverterController ===
class DummyArgs:
    def __init__(self, gremlin=None, filepath=None, url=None, age=False):
        self.gremlin = gremlin
        self.filepath = filepath
        self.url = url
        self.age = age


class TestGremlinConverterController(unittest.TestCase):
    def setUp(self):
        # Patch out the real conversion so that tests run fast.
        self.convert_patch = patch(
            "g2c.main.GremlinToCypherConverter.convert",
            return_value="MATCH (n) RETURN n",
        )
        self.mock_convert = self.convert_patch.start()
        # Patch the cache manager methods
        self.cache_patch = patch(
            "g2c.main.CacheManager.get_search_result", return_value=None
        )
        self.mock_get = self.cache_patch.start()
        self.cache_add_patch = patch("g2c.main.CacheManager.add_search_result")
        self.mock_add = self.cache_add_patch.start()

    def tearDown(self):
        self.convert_patch.stop()
        self.cache_patch.stop()
        self.cache_add_patch.stop()

    def test_read_code_from_filepath_success(self):
        # Simulate a file input.
        dummy_code = 'print("g.test")'
        args = DummyArgs(filepath="dummy.py")
        controller = GremlinConverterController(args)
        # Patch os.path.exists and open.
        with (
            patch("os.path.exists", return_value=True),
            patch("builtins.open", mock_open(read_data=dummy_code)),
        ):
            code, path = controller._read_code()
            self.assertEqual(code, dummy_code)
            self.assertEqual(path, "dummy.py")

    def test_read_code_from_filepath_not_found(self):
        args = DummyArgs(filepath="nofile.py")
        controller = GremlinConverterController(args)
        with patch("os.path.exists", return_value=False):
            with self.assertRaises(SystemExit) as cm:
                controller._read_code()
            self.assertIn("File not found", str(cm.exception))

    def test_read_code_from_url_success(self):
        dummy_code = "print('g.from_url')"
        args = DummyArgs(url="http://dummy.url")
        controller = GremlinConverterController(args)
        fake_response = MagicMock()
        fake_response.read.return_value = dummy_code.encode("utf-8")
        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_urlopen.return_value.__enter__.return_value = fake_response
            code, path = controller._read_code()
            self.assertEqual(code, dummy_code)
            self.assertEqual(path, "http://dummy.url")

    def test_read_code_from_gremlin_argument(self):
        # Direct query provided via --gremlin.
        args = DummyArgs(gremlin="“g.directquery”")  # using smart quotes
        controller = GremlinConverterController(args)
        code, path = controller._read_code()
        # The smart quotes should be replaced with standard quotes and then stripped.
        self.assertEqual(code, "g.directquery")
        self.assertEqual(path, "direct query")

    def test_read_code_none_provided(self):
        # When no valid input is provided, _read_code exits.
        args = DummyArgs()
        controller = GremlinConverterController(args)
        with self.assertRaises(SystemExit) as cm:
            controller._read_code()
        self.assertIn("No valid input provided", str(cm.exception))

    def test_process_no_extracted_queries(self):
        # Test process branch where extraction returns empty list.
        args = DummyArgs(filepath="dummy.py")
        controller = GremlinConverterController(args)
        # Patch _read_code to return dummy code that produces an empty extraction result,
        # and then patch QueryExtractor.extract() to return [].
        with patch.object(
            GremlinConverterController,
            "_read_code",
            return_value=("dummy code", "dummy.py"),
        ):
            with patch.object(QueryExtractor, "extract", return_value=[]):
                with self.assertRaises(SystemExit) as cm:
                    controller.process()
                self.assertIn("No Gremlin queries found", str(cm.exception))

    def test_process_success(self):
        # Test process method when conversion returns nonempty query.
        args = DummyArgs(gremlin="g.V()")
        controller = GremlinConverterController(args)
        # In direct-query branch, _read_code wraps into a list.
        # process() will then call converter.convert (our patched version returns "MATCH (n) RETURN n")
        with patch("builtins.print") as m_print:
            controller.process()
            # Check that the printed output contains the conversion.
            printed = "".join(call.args[0] for call in m_print.call_args_list)
            self.assertIn("MATCH (n) RETURN n", printed)

    def test_format_cypher_failure(self):
        # Test format_cypher returns red [Failed] if empty string provided.
        args = DummyArgs(gremlin="g.V()", age=False)
        controller = GremlinConverterController(args)
        result = controller.format_cypher("")
        self.assertIn("[Failed]", result)

    def test_format_cypher_for_age(self):
        # Test format_cypher for apache AGE.
        args = DummyArgs(gremlin="g.V()", age=True)
        controller = GremlinConverterController(args)
        # To test the static method format_for_age, we patch the ANTLR parts.
        # We simulate that the visitor.visit returns None so that format_for_age returns
        # a SELECT statement.
        with (
            patch("g2c.main.CypherLexer") as mock_lexer,
            patch("g2c.main.CypherParser") as mock_parser,
            patch("g2c.main.MyCypherVisitor") as mock_visitor,
        ):
            # Fake parser and visitor behavior.
            fake_visitor = MagicMock()
            fake_visitor.visit.return_value = None
            fake_visitor.expressions = ["p", "q"]
            mock_visitor.return_value = fake_visitor

            formatted = controller.format_cypher("MATCH (n) RETURN n")
            # Should contain a SELECT statement, with agtype types.
            self.assertIn("SELECT * FROM cypher", formatted)
            self.assertIn("p agtype", formatted)
            self.assertIn("q agtype", formatted)


# === Test MyCypherVisitor ===
class DummyCtx:
    def __init__(self, text):
        self._text = text

    def getText(self):
        return self._text

    # Return no children to avoid traversing.
    def getChildCount(self):
        return 0

    # Even if getChild were needed, it should return a DummyCtx with an accept method.
    def getChild(self, i):
        # For testing, you might return self or a new DummyCtx.
        return self

    # Implement the required accept() method.
    def accept(self, visitor):
        # Calling the generic visit for the dummy node.
        return visitor.visit(self)


class TestMyCypherVisitor(unittest.TestCase):
    def test_visitOC_Expression_adds_valid_identifier(self):
        visitor = MyCypherVisitor()
        # Use a dummy context whose text is a valid identifier.
        ctx = DummyCtx("abc123")
        visitor.visitOC_Expression(ctx)
        self.assertIn("abc123", visitor.expressions)

    def test_visitOC_Expression_ignores_invalid_identifier(self):
        visitor = MyCypherVisitor()
        ctx = DummyCtx("123abc")  # does not match the regex (must start with letter)
        visitor.visitOC_Expression(ctx)
        self.assertNotIn("123abc", visitor.expressions)


# === Test main and argument parsing ===
class TestMain(unittest.TestCase):
    @patch("g2c.main.parse_arguments")
    @patch("g2c.main.GremlinConverterController")
    def test_main_calls_controller(self, mock_controller_class, mock_parse_args):
        dummy_args = argparse.Namespace(
            gremlin="g.V()", filepath=None, url=None, age=False
        )
        mock_parse_args.return_value = dummy_args
        # Call main() (it will instantiate a GremlinConverterController)
        main()
        mock_controller_class.assert_called_once_with(dummy_args)
        instance = mock_controller_class.return_value
        instance.process.assert_called_once()

    def test_parse_arguments_required(self):
        # Test that argparse requires one of the mutually exclusive groups.
        testargs = ["prog", "-g", "g.V()"]
        with patch.object(sys, "argv", testargs):
            args = parse_arguments()
            self.assertEqual(args.gremlin, "g.V()")
            self.assertFalse(args.age)

    def test_parse_arguments_age_flag(self):
        # Test the age flag is parsed correctly.
        testargs = ["prog", "-g", "g.V()", "--age"]
        with patch.object(sys, "argv", testargs):
            args = parse_arguments()
            self.assertTrue(args.age)


if __name__ == "__main__":
    unittest.main()
