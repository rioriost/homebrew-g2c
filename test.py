#!/usr/bin/env python3
import argparse
import io
import json
import os
import sys
import unittest
from contextlib import contextmanager
from unittest import mock

# Import the module under test.
# (Make sure your PYTHONPATH is set appropriately so that src/ is importable.)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from g2c.main import (
    QueryExtractor,
    CacheManager,
    GremlinToCypherConverter,
    DryRunner,
    GremlinConverterController,
    parse_arguments,
    main,
)


# A dummy OpenAI client for testing the GremlinToCypherConverter.
class DummyChatCompletion:
    def __init__(self, content):
        self.message = mock.Mock(content=content)


class DummyChat:
    def __init__(self, content):
        self.content = content

    def create(self, messages, model, temperature):
        # Return a dummy completions object with the desired text.
        # For testing, return a fixed cypher query.
        dummy_choice = mock.Mock(message=mock.Mock(content="MATCH (n) RETURN n"))
        dummy = mock.Mock(choices=[dummy_choice])
        return dummy


class DummyOpenAI:
    def __init__(self, api_key):
        self.api_key = api_key
        self.chat = DummyChat(content="dummy")


# A dummy connection object for DryRunner.
class DummyConnection:
    def __init__(self, should_fail=False):
        self.should_fail = should_fail

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def execute(self, query):
        if self.should_fail:
            raise RuntimeError("Execution failed")
        # else succeed silently


# A dummy connection pool for DryRunner.
class DummyConnectionPool:
    def __init__(self, dsn):
        self.dsn = dsn

    def open(self):
        pass

    def connection(self):
        # Return a dummy connection; let the caller decide if it fails.
        return DummyConnection()

    def close(self):
        pass


# Helper context manager to capture printed output.
@contextmanager
def captured_stdout():
    new_out = io.StringIO()
    old = sys.stdout
    sys.stdout = new_out
    try:
        yield sys.stdout
    finally:
        sys.stdout = old


class TestQueryExtractor(unittest.TestCase):
    def test_extract_python_literal(self):
        # Code that does not import gremlin_python.
        code = 'x = "g.foo.query"'
        extractor = QueryExtractor(code, "dummy.py")
        results = extractor.extract()
        # tokenize method should pick up the literal on line 1.
        self.assertTrue(any("g.foo.query" in lit for _, lit in results))

    def test_extract_gremlin_queries_ast(self):
        # Code that imports gremlin_python so that AST extraction is used.
        # Build a simple call: g.V().has("name", "john")
        code = """
from gremlin_python.driver import client
def test():
    return g.V().has("name", "john")
"""
        extractor = QueryExtractor(code, "dummy.py")
        results = extractor.extract()
        # Expect at least one tuple with a non-negative line number.
        self.assertTrue(len(results) > 0)
        line, query = results[0]
        self.assertTrue(line >= 1)
        self.assertIn("g.V()", query)

    def test_extract_generic_literals(self):
        # For non-.py extension file use regex extraction.
        code = """
// some comment "g.ignore"
var query = "g.someQuery";
"""
        extractor = QueryExtractor(code, "dummy.java")
        results = extractor.extract()
        self.assertEqual(len(results), 1)
        lineno, literal = results[0]
        self.assertIn("g.someQuery", literal)

    def test_tokenize_error(self):
        # Provide code that triggers an exception within tokenize.
        bad_code = "def foo(:"
        extractor = QueryExtractor(bad_code, "dummy.py")
        with self.assertRaises(SystemExit) as cm:
            extractor._extract_gremlin_from_literals()
        self.assertIn("Error while parsing Python file", str(cm.exception))

    def test_ast_parse_error(self):
        # Provide invalid Python code to trigger a parse error.
        bad_code = "def foo(:"
        extractor = QueryExtractor(bad_code, "dummy.py")
        with self.assertRaises(SystemExit) as cm:
            extractor._extract_gremlin_queries()
        self.assertIn("Error while parsing Python file", str(cm.exception))


class TestCacheManager(unittest.TestCase):
    def setUp(self):
        # Patch os.path.exists to simulate that the cache file already exists.
        self.cache_file = os.path.join(os.path.expanduser("~"), ".g2c_cache")
        patcher = mock.patch("g2c.main.os.path.exists", return_value=True)
        self.addCleanup(patcher.stop)
        self.mock_exists = patcher.start()

        # Patch open to simulate reading/writing a cache
        self.initial_cache = {"foo": "bar"}
        self.mock_open = mock.mock_open(read_data=json.dumps(self.initial_cache))
        patcher2 = mock.patch("g2c.main.open", self.mock_open, create=True)
        self.addCleanup(patcher2.stop)
        patcher2.start()

    def test_load_cache(self):
        cm = CacheManager()
        self.assertEqual(cm.cache, self.initial_cache)

    def test_add_get_search_result(self):
        cm = CacheManager()
        # Add new search result.
        cm.add_search_result("query1", "cypher1")
        # Now get the search result.
        result = cm.get_search_result("query1")
        self.assertEqual(result, "cypher1")


class TestGremlinToCypherConverter(unittest.TestCase):
    def setUp(self):
        # Set API key for testing.
        os.environ[GremlinToCypherConverter.KEY_ENV_NAME] = "dummy_key"
        # Patch the OpenAI client to use our dummy.
        self.patcher = mock.patch("g2c.main.OpenAI", DummyOpenAI)
        self.patcher.start()

    def tearDown(self):
        self.patcher.stop()

    def test_convert_success(self):
        converter = GremlinToCypherConverter()
        result = converter.convert("g.V()")
        # Our DummyChat.create returns a fixed cypher query.
        self.assertEqual(result, "MATCH (n) RETURN n")

    def test_convert_failure(self):
        # Force an exception inside the converter.
        converter = GremlinToCypherConverter()
        # Patch client's chat.completions.create to throw an exception.
        with mock.patch.object(
            converter.client.chat,
            "create",
            side_effect=RuntimeError(
                "'DummyChat' object has no attribute 'completions'"
            ),
        ):
            with mock.patch("g2c.main.print") as mock_print:
                result = converter.convert("g.V()")
                self.assertEqual(result, "")
                mock_print.assert_called_with(
                    "Error during conversion: 'DummyChat' object has no attribute 'completions'"
                )

    def test_missing_api_key(self):
        # Remove API key and test that __init__ exits.
        del os.environ[GremlinToCypherConverter.KEY_ENV_NAME]
        with self.assertRaises(SystemExit) as cm:
            GremlinToCypherConverter()
        self.assertIn("API key not found", str(cm.exception))
        os.environ[GremlinToCypherConverter.KEY_ENV_NAME] = "dummy_key"


class TestDryRunner(unittest.TestCase):
    def setUp(self):
        # Set environment variable required for DryRunner.
        os.environ["PG_CONNECTION_STRING"] = "postgresql://dummy"
        # Patch ConnectionPool to use our dummy pool.
        self.patcher = mock.patch("g2c.main.ConnectionPool", DummyConnectionPool)
        self.patcher.start()

    def tearDown(self):
        self.patcher.stop()

    def test_run_success(self):
        runner = DryRunner()
        msg = runner.run("MATCH (n) RETURN n")
        self.assertIn("[Query executed successfully]", msg)

    def test_run_failure(self):
        # Patch the connection to raise an error.
        pool = DummyConnectionPool("dummy")
        with mock.patch.object(
            pool, "connection", return_value=DummyConnection(should_fail=True)
        ):
            runner = DryRunner()
            # Replace runner.pool with our patched pool.
            runner.pool = pool
            msg = runner.run("MATCH (n) RETURN n")
            self.assertIn("[Error executing query: ", msg)


class TestGremlinConverterController(unittest.TestCase):
    def setUp(self):
        # Set a dummy API key needed by the controller.
        os.environ[GremlinToCypherConverter.KEY_ENV_NAME] = "dummy_key"
        # Create a dummy argparse.Namespace object.
        self.args = argparse.Namespace(
            age=False,
            model="dummy-model",
            dryrun=False,
            gremlin="g.V().has('name','john')",
            filepath=None,
            url=None,
        )
        # Patch CacheManager and GremlinToCypherConverter inside the controller.
        self.cache_manager_patch = mock.patch("g2c.main.CacheManager")
        self.converter_patch = mock.patch("g2c.main.GremlinToCypherConverter")
        self.mock_cache_manager_class = self.cache_manager_patch.start()
        self.mock_converter_class = self.converter_patch.start()
        # Make sure the conversion returns a fixed string.
        self.mock_converter_instance = mock.Mock()
        self.mock_converter_instance.convert.return_value = "MATCH (n) RETURN n"
        self.mock_converter_class.return_value = self.mock_converter_instance
        # For dryrun tests, patch DryRunner.
        self.dryrunner_patch = mock.patch("g2c.main.DryRunner")
        self.mock_dryrunner_class = self.dryrunner_patch.start()
        self.addCleanup(self.cache_manager_patch.stop)
        self.addCleanup(self.converter_patch.stop)
        self.addCleanup(self.dryrunner_patch.stop)
        # Patch _read_code to return the gremlin query.
        self.read_code_patch = mock.patch.object(
            GremlinConverterController,
            "_read_code",
            return_value=(self.args.gremlin, "direct query"),
        )
        self.read_code_patch.start()
        self.addCleanup(self.read_code_patch.stop)

    def test_process_direct_query(self):
        # Testing process() when input is a direct gremlin query.
        controller = GremlinConverterController(self.args)
        # In this case, _read_code returns a direct query so extraction wraps it in a list.
        with captured_stdout() as out:
            controller.process()
        output = out.getvalue()
        self.assertIn("Converted Cypher queries:", output)
        self.assertIn("MATCH (n) RETURN n", output)
        # Verify that conversion and cache were called.
        self.mock_converter_instance.convert.assert_called_with(self.args.gremlin)

    def test_format_cypher_failed(self):
        controller = GremlinConverterController(self.args)
        # Test when conversion returns empty string.
        res = controller.format_cypher("")
        self.assertIn("[Failed]", res)

    def test_format_cypher_age_without_dryrun(self):
        self.args.age = True
        controller = GremlinConverterController(self.args)
        # Simulate a conversion result that includes a return clause.
        cypher = "MATCH (n) RETURN n.name, n.age"
        res = controller.format_cypher(cypher)
        # Should contain call to cypher with stored procedure or standard cypher for AGE.
        self.assertTrue("cypher('GRAPH_NAME'" in res)

    def test_format_cypher_age_with_dryrun(self):
        self.args.age = True
        self.args.dryrun = True
        # Create a dummy runner that returns a known result.
        dummy_runner = mock.Mock()
        dummy_runner.run.return_value = "[Dummy execution]"
        controller = GremlinConverterController(self.args)
        controller.runner = dummy_runner
        cypher = "MATCH (n) RETURN n"
        res = controller.format_cypher(cypher)
        self.assertIn("[Dummy execution]", res)

    def test_format_for_age_and_extract_return_values(self):
        # Test the two connected helper methods.
        controller = GremlinConverterController(self.args)
        cypher_query = "MATCH (n) RETURN n.age, n.name"
        ret_vals = controller.extract_return_values(cypher_query)
        self.assertIn("n", ret_vals)  # since 'n.name' and 'n.age' are trimmed to 'n'
        formatted = controller.format_for_age(cypher_query)
        self.assertIn("cypher('GRAPH_NAME'", formatted)


class TestParseArgumentsMain(unittest.TestCase):
    def test_parse_arguments_gremlin(self):
        test_argv = ["prog", "-g", "g.V()"]
        with mock.patch("sys.argv", test_argv):
            args = parse_arguments()
            self.assertEqual(args.gremlin, "g.V()")
            self.assertFalse(args.age)
            self.assertFalse(args.dryrun)
            self.assertEqual(args.model, "gpt-4o-mini")

    def test_parse_arguments_filepath(self):
        test_argv = ["prog", "-f", "dummy.py"]
        with mock.patch("sys.argv", test_argv):
            args = parse_arguments()
            self.assertEqual(args.filepath, "dummy.py")
            self.assertIsNone(args.gremlin)
            self.assertIsNone(args.url)

    def test_parse_arguments_url(self):
        test_argv = ["prog", "-u", "http://example.com/code.py"]
        with mock.patch("sys.argv", test_argv):
            args = parse_arguments()
            self.assertEqual(args.url, "http://example.com/code.py")
            self.assertIsNone(args.gremlin)
            self.assertIsNone(args.filepath)

    def test_main_calls_controller_process(self):
        # Test main() by intercepting the controller instance.
        test_argv = ["prog", "-g", "g.V()"]
        with mock.patch("sys.argv", test_argv):
            with mock.patch(
                "src.g2c.main.GremlinConverterController"
            ) as mock_controller_class:
                instance = mock_controller_class.return_value
                main()
                instance.process.assert_called_once()


if __name__ == "__main__":
    unittest.main()
