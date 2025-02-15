import argparse
import unittest
from unittest.mock import patch, mock_open, MagicMock
import os
import json
from io import StringIO

# Import the functions and classes from the main module
from g2c.main import SearchCache, convert_gremlin_to_cypher, main


class TestSearchCache(unittest.TestCase):
    @patch("os.path.exists", return_value=True)
    @patch(
        "builtins.open",
        new_callable=mock_open,
        read_data='{"test_query": "test_result"}',
    )
    def test_load_cache(self, mock_open, mock_exists):
        cache = SearchCache()
        self.assertEqual(cache.cache, {"test_query": "test_result"})

    @patch("os.path.exists", return_value=False)
    @patch("builtins.open", new_callable=mock_open)
    def test_load_cache_no_file(self, mock_open, mock_exists):
        cache = SearchCache()
        self.assertEqual(cache.cache, {})

    @patch("builtins.open", new_callable=mock_open, read_data="{}")
    def test_save_cache(self, mock_open):
        cache = SearchCache()
        cache.cache = {"test_query": "test_result"}
        cache._save_cache()
        mock_open().write.assert_called()  # Ensure write was called
        written_data = "".join(
            call.args[0] for call in mock_open().write.call_args_list
        )
        self.assertEqual(
            written_data,
            json.dumps({"test_query": "test_result"}, ensure_ascii=False, indent=4),
        )

    @patch("builtins.open", new_callable=mock_open, read_data="{}")
    def test_add_search_result(self, mock_open):
        cache = SearchCache()
        cache.add_search_result("new_query", "new_result")
        self.assertEqual(cache.cache, {"new_query": "new_result"})
        mock_open().write.assert_called()  # Ensure write was called
        written_data = "".join(
            call.args[0] for call in mock_open().write.call_args_list
        )
        self.assertEqual(
            written_data,
            json.dumps({"new_query": "new_result"}, ensure_ascii=False, indent=4),
        )

    @patch(
        "builtins.open",
        new_callable=mock_open,
        read_data='{"test_query": "test_result"}',
    )
    def test_get_search_result(self, mock_open):
        cache = SearchCache()
        result = cache.get_search_result("test_query")
        self.assertEqual(result, "test_result")


class TestConvertGremlinToCypher(unittest.TestCase):
    @patch("openai.OpenAI")
    def test_convert_gremlin_to_cypher(self, mock_openai):
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(message=MagicMock(content="MATCH (n) RETURN n"))
        ]
        mock_client.chat.completions.create.return_value = mock_response

        result = convert_gremlin_to_cypher("g.V()")
        self.assertEqual(result, "MATCH (n) RETURN n")

    @patch.dict(os.environ, {}, clear=True)
    def test_convert_gremlin_to_cypher_no_api_key(self):
        with self.assertRaises(SystemExit):
            convert_gremlin_to_cypher("g.V()")

    @patch.dict(os.environ, {"G2C_OPENAI_API_KEY": "fake_api_key"})
    @patch("openai.OpenAI")
    def test_convert_gremlin_to_cypher_exception(self, mock_openai):
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        mock_client.chat.completions.create.side_effect = Exception("API error")

        result = convert_gremlin_to_cypher("g.V()")
        self.assertEqual(result, "")


class TestMain(unittest.TestCase):
    @patch(
        "argparse.ArgumentParser.parse_args",
        return_value=argparse.Namespace(gremlin_query="g.V()"),
    )
    @patch("builtins.open", new_callable=mock_open, read_data="{}")
    @patch("src.g2c.main.convert_gremlin_to_cypher", return_value="MATCH (n) RETURN n")
    @patch("sys.stdout", new_callable=StringIO)
    def test_main(self, mock_stdout, mock_convert, mock_open, mock_args):
        main()
        self.assertIn("MATCH (n) RETURN n", mock_stdout.getvalue())

    @patch(
        "argparse.ArgumentParser.parse_args",
        return_value=argparse.Namespace(gremlin_query="g.V()"),
    )
    @patch(
        "builtins.open",
        new_callable=mock_open,
        read_data='{"g.V()": "MATCH (n) RETURN n"}',
    )
    @patch("sys.stdout", new_callable=StringIO)
    def test_main_cache_hit(self, mock_stdout, mock_open, mock_args):
        main()
        self.assertIn("MATCH (n) RETURN n", mock_stdout.getvalue())


if __name__ == "__main__":
    unittest.main()
