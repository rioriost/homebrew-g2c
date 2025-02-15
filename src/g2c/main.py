#!/usr/bin/env python3
import argparse
import os
import sys
from openai import OpenAI
import logging
import json

logging.basicConfig(level=logging.ERROR)


class SearchCache:
    def __init__(self):
        self.cache_file = os.path.join(os.path.expanduser("~"), ".g2c_cache")
        self.cache = self._load_cache()

    def _load_cache(self):
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "r") as file:
                return json.load(file)
        return {}

    def _save_cache(self):
        with open(self.cache_file, "w") as file:
            json.dump(self.cache, file, ensure_ascii=False, indent=4)

    def add_search_result(self, keyword: str, result: str):
        self.cache[keyword] = result
        self._save_cache()

    def get_search_result(self, keyword: str) -> str:
        return self.cache.get(keyword, None)


def convert_gremlin_to_cypher(gremlin_query: str) -> str:
    """
    Converts a Gremlin query into a Cypher query using the OpenAI API.

    Parameters:
        gremlin_query (str): The Gremlin query to convert.

    Returns:
        str: The converted Cypher query.
    """
    KEY_ENV_NAME = "G2C_OPENAI_API_KEY"
    try:
        api_key = os.environ[KEY_ENV_NAME]
    except KeyError:
        sys.exit(
            f"API key not found. Please set the {KEY_ENV_NAME} environment variable."
        )

    client = OpenAI(api_key=api_key)

    # Create the prompt for conversion by including the Gremlin query.
    prompt = f"""
    Convert the following Gremlin query to an equivalent Cypher query:

    {gremlin_query}
    """

    try:
        # Call the Azure OpenAI Chat Completion API with a system message and the user prompt.
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant that converts Gremlin queries into Cypher queries. You reply just a converted Cypher query without quotes.",
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
    parser = argparse.ArgumentParser(
        description="Convert Gremlin queries to Cypher queries."
    )
    parser.add_argument("gremlin_query", type=str, help="The Gremlin query to convert.")
    args = parser.parse_args()

    logging.debug(f"{args.gremlin_query=}")

    search_cache = SearchCache()
    query = args.gremlin_query.replace("“", '"').replace("”", '"')
    if cypher_result := search_cache.get_search_result(query):
        print(cypher_result)
    else:
        # Perform the conversion.
        cypher_query = convert_gremlin_to_cypher(query)
        search_cache.add_search_result(query, cypher_query)
        print(cypher_query)


if __name__ == "__main__":
    main()
