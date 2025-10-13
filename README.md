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

## Overview

- Supervisor agent coordinates retrieval/generation based on chat state.
- MongoDB Atlas stores graph topology plus content chunks and embeddings.
- Gemini models (via Vertex/Gemini API keys) provide reasoning, summarisation, and answer generation.
- `doit` tasks wrap typing, linting, testing, and coverage routines for local development.

## Project Structure

```text
|- src
|   |-- chat_agh
|   |      |- agents
|   |      |    |- retrieval
|   |      |    |    |- context_retrieval.py
|   |      |    |    |- similarity_search.py
|   |      |    |    |- summary_generation.py
|   |      |    |    |- utils.py
|   |      |    |- generation_agent.py
|   |      |    |- retrieval_agent.py
|   |      |    |- supervisor_agent.py
|   |      |- nodes
|   |      |    |- generation_node.py
|   |      |    |- initial_retrieval_node.py
|   |      |    |- supervisor_node.py
|   |      |    |- retrieval_node.py
|   |      |- utils
|   |      |    |- agents_info.py
|   |      |    |- chat_history.py
|   |      |    |- utils.py
|   |      |- vector_store
|   |      |    |- mongodb.py
|   |      |    |- utils.py
|   |      |- graph.py
|   |      |- prompts.py
|   |      |- states.py
```
## Core Components:
- **Supervisor Agent**: Oversees the response generation workflow, decides which agents should handle a given query, and ensures coherent coordination between retrieval and generation.
- **Retrieval Agents**: Extract relevant information from vector collections and traverse the knowledge graph to gather additional context, enabling more informed responses.
- **Generation Agent**: Integrates all retrieved knowledge, including graph-derived context, to produce the final, human-readable answer.

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
