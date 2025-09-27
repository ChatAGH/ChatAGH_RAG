## Code structure
```
|- docs
|   |- architecture.md
|   |- developer_guide.md
|   |- overview.md
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
|- .env.template
|- dodo.py
|- poetry.lock
|- pyproject.toml
|- README.md 
```
## Flow diagram

## Core Components:
- **Supervisor Agent**: Oversees the response generation workflow, decides which agents should handle a given query, and ensures coherent coordination between retrieval and generation.
- **Retrieval Agents**: Extract relevant information from vector collections and traverse the knowledge graph to gather additional context, enabling more informed responses.
- **Generation Agent**: Integrates all retrieved knowledge, including graph-derived context, to produce the final, human-readable answer.
