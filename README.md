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
usage: g2c [-h] gremlin_query

Convert Gremlin queries to Cypher queries.

positional arguments:
  gremlin_query  The Gremlin query to convert.

options:
  -h, --help     show this help message and exit
```

The indentical usage is shown below.

```bash
g2c 'g.V().has(“name”, “Alice”).as(“a”).V().has(“name”, “Bob”).as(“b”).select(“a”, “b”).by(“name”)'
MATCH (a {name: "Alice"}), (b {name: "Bob"}) RETURN a.name AS a, b.name AS b
```

## Release Notes

### 0.1.0 Release
* Initial release.

## License
MIT License
