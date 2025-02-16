# g2c

![License](https://img.shields.io/badge/license-MIT-blue.svg)

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
usage: g2c [-h] (-g GREMLIN | -f FILEPATH | -u URL)

Convert Gremlin queries to Cypher queries.

options:
  -h, --help            show this help message and exit
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
g2c --url https://raw.githubusercontent.com/nedlowe/gremlin-python-example/refs/heads/master/app.py
Converted Cypher queries:

line 42, g.V(person_id).toList() ->
MATCH (p) WHERE id(p) = person_id RETURN p

line 42, g.V(person_id) ->
MATCH (n) WHERE id(n) = person_id RETURN n

line 55, g.V(vertex).valueMap().toList() ->
MATCH (n) WHERE id(n) = $vertex RETURN n

line 55, g.V(vertex).valueMap() ->
MATCH (n) WHERE id(n) = $vertex RETURN n

line 55, g.V(vertex) ->
MATCH (n) WHERE id(n) = vertex RETURN n

line 77, g.addV('person').property(T.id, person_id).next() ->
CREATE (n:person {id: person_id})
......
```

with -f(--filepath)

```bash
g2c --file ~/Desktop/g.py
Converted Cypher queries:

line 1, "g.V()" ->
MATCH (n) RETURN n

line 3, "g.E()" ->
MATCH ()-[r]->() RETURN r

line 4, "g.V('vertexId')" ->
MATCH (n) WHERE id(n) = 'vertexId' RETURN n
......
```

## Release Notes

### 0.2.0 Release
* Changed the default behaviour to accept a Gremlin query (-g), a file (-f), or a URL (-u).
* Added feature to extract Gremlin queries from a file or URL.
* Added CacheManager and a feature to deploy a cache file to user home directory if it doesn't exist.

### 0.1.0 Release
* Initial release.

## License
MIT License
