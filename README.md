# g2c

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/Python-3.11%2B-blue)

## Overview

g2c is a python script to convert Gremlin query to Cypher query with OpenAI API

## Table of Contents

- [Installation](#installation)
- [Prerequisites](#prerequisites)
- [Usage](#usage)
- [Release Notes](#release-notes)
- [License](#license)

## Installation

Just add tap and install homebrew package.

```bash
brew tap rioriost/g2c
brew install g2c
```

## Prerequisites

- Python 3.11 or higher
- [OpenAI API key](https://platform.openai.com/account/api-keys) enabled to call gpt-4o-mini

## Usage

Execute g2c command.

```bash
g2c --help
usage: g2c [-h] [-a] [-m MODEL] [-d] (-g GREMLIN | -f FILEPATH | -u URL)

Convert Gremlin queries to Cypher queries.

options:
  -h, --help            show this help message and exit
  -a, --age             Convert to the Cypher query for Apache AGE.
  -m MODEL, --model MODEL
                        OpenAI model to use.
  -d, --dryrun          Dry run with PostgreSQL. Requires a valid PostgreSQL connection string as 'PG_CONNECTION_STRING' environment variable.
  -g GREMLIN, --gremlin GREMLIN
                        The Gremlin query to convert.
  -f FILEPATH, --filepath FILEPATH
                        Path to the source code file (.py, .java, .cs, .txt)
  -u URL, --url URL     URL to the source code file (.py, .java, .cs, .txt)
```

The indentical usage is shown below.

with -g(--gremlin)

```bash
g2c -g 'g.V().has(“name”, “Alice”).as(“a”).V().has(“name”, “Bob”).as(“b”).select(“a”, “b”).by(“name”)'
Converted Cypher queries:

line 1, g.V().has("name", "Alice").as("a").V().has("name", "Bob").as("b").select("a", "b").by("name") ->
MATCH (a {name: "Alice"}), (b {name: "Bob"}) RETURN a.name AS a, b.name AS b
```

with -u(--url)

```bash
g2c -u https://raw.githubusercontent.com/nedlowe/gremlin-python-example/refs/heads/master/app.py
Converted Cypher queries:

line 42, g.V(person_id).toList() ->
MATCH (n) WHERE id(n) = $person_id RETURN n

line 42, g.V(person_id) ->
MATCH (n) WHERE id(n) = $person_id RETURN n

line 55, g.V(vertex).valueMap().toList() ->
MATCH (n) WHERE ID(n) = $vertex RETURN properties(n)

line 55, g.V(vertex).valueMap() ->
MATCH (n) WHERE ID(n) = $vertex RETURN properties(n)
......
```

with -f(--filepath)

```bash
g2c -f ~/Desktop/gremlin_samples.py
Converted Cypher queries:

line 1, g.V() ->
MATCH (n) RETURN n

line 2, g.E() ->
MATCH ()-[r]-() RETURN r

line 3, g.V().hasLabel('person') ->
MATCH (n:person) RETURN n

line 4, g.V().hasLabel('software') ->
MATCH (n:software) RETURN n
......
```

with -a(--age)

```bash
g2c -a -g "g.V().hasLabel('person').aggregate('a')"
Converted Cypher queries:

line 1, g.V().hasLabel('person').aggregate('a') ->
SELECT * FROM cypher('GRAPH_NAME', $$ MATCH (n:person) WITH collect(n) AS a RETURN a $$) AS (a agtype);
```

with -d(--dryrun)

```bash
g2c -d -g "g.V().hasLabel('person').aggregate('a')"
Converted Cypher queries:

line 1, g.V().hasLabel('person').aggregate('a') ->
SELECT * FROM cypher('GRAPH_NAME', $$ MATCH (n:person) WITH collect(n) AS a RETURN a $$) AS (a agtype);
[Query executed successfully]
```

```bash
g2c -d -g "g.V(person).property(prop_name, prop_value)"
Converted Cypher queries:

line 1, g.V(person).property(prop_name, prop_value) ->
DEALLOCATE ALL; PREPARE cypher_stored_procedure(agtype) AS SELECT * FROM cypher('GRAPH_NAME', $$ MATCH (n) WHERE ID(n) = $person SET n[$prop_name] = $prop_value RETURN n $$, $1) AS (n agtype);EXECUTE cypher_stored_procedure('{"person": 12345, "prop_name": 12345, "prop_value": 12345}');
[Error executing query: SET clause expects a property name
LINE 1: ...APH_NAME', $$ MATCH (n) WHERE ID(n) = $person SET n[$prop_na...
                                                             ^]
```

## Release Notes

### 0.4.3 Release
* Updated for the dependencies

### 0.4.2 Release
* Fixed a small bug

### 0.4.1 Release
* Updated for the dependencies

### 0.4.0 Release
* Stopped to use antlr to analyze the Cypher query, removed the dependencies to antlr and generated lexer, parser, and visitor.
* Added '--model' argument to enable switching the default model, 'gpt-4o-mini' to others such as 'o3-mini' and so on.
  If you have an OpenAI subscription to use smarter models, strongly recommended to use them.
* Added '--dryrun' argument to enable dry run mode. If dryrun is true, this script will try to connect to PostgreSQL with Apache AGE extension and to execute the converted Cypher query.
  It requires PG_CONNECTION_STRING environment variable for 'TESTING' PostgreSQL.

### 0.3.0 Release
* Refactored the code
* Fixed a bug to save a cache. If you're using the old versions, please delete the .g2c_cache file under your home directory.
* Added '--age' argument to convert Gremlin query to Cypher using Apache AGE

### 0.2.0 Release
* Changed the default behaviour to accept a Gremlin query (-g), a file (-f), or a URL (-u).
* Added feature to extract Gremlin queries from a file or URL.
* Added CacheManager and a feature to deploy a cache file to user home directory if it doesn't exist.

### 0.1.0 Release
* Initial release.

## License
MIT License
