from typing import TypedDict

from src.utils.agents_info import AgentsInfo
from src.utils.chat_history import ChatHistory


class ChatState(TypedDict):
    chat_history: ChatHistory
    agents_info: AgentsInfo
    retrieval_decision: bool
    agents_queries: dict
    response: str


class RetrievalState(TypedDict):
    query: str
    retrieved_chunks: dict
    retrieved_context: dict
    summary: str
