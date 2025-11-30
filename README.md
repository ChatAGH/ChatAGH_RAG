<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
**Table of Contents**  *generated with [DocToc](https://github.com/thlorenz/doctoc)*

- [ChatAGH Core](#chatagh-core)
  - [Overview](#overview)
  - [Project Structure](#project-structure)
  - [Core Components:](#core-components)
  - [Requirements](#requirements)
  - [Installation](#installation)
  - [Configuration](#configuration)
  - [Testing](#testing)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

# ChatAGH Core

This project implements an advanced **Agentic Retrieval-Augmented Generation (RAG)** system, serving as the backend for the **ChatAGH** project. Its primary goal is to generate contextually relevant, accurate, and meaningful responses to user queries by combining the strengths of LLM agents with **graph-based** knowledge retrieval.

Unlike standard RAG systems that rely solely on vector-based retrieval, this project leverages both vector and graph-structured data to enrich the context. By modeling relationships between e
ntities and concepts in a knowledge graph, the system can:
- Provide deeper contextual understanding for complex queries
- Connect related pieces of information that may be distributed across multiple sources
- Enhance reasoning by capturing dependencies and hierarchies within the data

## Requirements

- Python 3.11+
- Poetry for dependency management
- Access to Gemini API keys (Vertex or Google Generative AI)
- MongoDB Atlas cluster with vector and search capabilities

## Installation

```bash
poetry install
```

## Configuration

Create `.env` file and copy `.env.example` to it:

 ```bash
    cp .env.example > .env
    ```

Ensure MongoDB collections contain:

- `chunks`: documents with text, metadata, and embedding fields.
- `edges`: `{source, target}` graph links for context expansion.

Recommended Atlas indexes:

- Search index (default) on the chunk `text` field.
- Vector index (`vector_index`) on the chunk `embedding` field.

## Development

Run the full checks with `doit`:

```bash
poetry run doit
```
This will run:
- Type checking (mypy)
- Code linting (ruff)
- Code formatting (ruff)

## Testing

The `tests/` directory is currently empty. Suggested additions:

- Unit tests for supervisor decision logic (mock Gemini responses).
- Vector store integration tests (with fake MongoDB or fixtures).
- Graph-level integration tests that mock retrieval/generation for deterministic assertions.
