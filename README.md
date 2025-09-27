# Graph-Augmented Agentic RAG for Web Data
This project implements an advanced **Agentic Retrieval-Augmented Generation (RAG)** system, serving as the backend for the **ChatAGH** project. Its primary goal is to generate contextually relevant, accurate, and meaningful responses to user queries by combining the strengths of LLM agents with **graph-based** knowledge retrieval.

Unlike standard RAG systems that rely solely on vector-based retrieval, this project leverages both vector and graph-structured data to enrich the context. By modeling relationships between entities and concepts in a knowledge graph, the system can:
- Provide deeper contextual understanding for complex queries
- Connect related pieces of information that may be distributed across multiple sources
- Enhance reasoning by capturing dependencies and hierarchies within the data

An overview of the project and its code structure is available in [docs/overview.md](https://github.com/ChatAGH/ChatAGH_RAG/docs/overview.md).

A detailed description of the system architecture can be found [docs/architecture.md](https://github.com/ChatAGH/ChatAGH_RAG/docs/architecture.md).

For instructions on configuring the environment and running the system, please refer to [docs/developer_guide.md](https://github.com/ChatAGH/ChatAGH_RAG/docs/developer_guide.md).

The project uses a pre-populated database. For more information on how the data was collected, processed, and stored, please see the https://github.com/ChatAGH/ChatAGH_DataCollecting repository.