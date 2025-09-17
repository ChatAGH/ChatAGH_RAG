from typing import TypedDict, Dict, Any

from langchain_core.documents import Document

from chat_agh.utils.agents_info import AgentsInfo
from chat_agh.utils.chat_history import ChatHistory


class ChatState(TypedDict, total=False):
    context: list[Document]
    chat_history: ChatHistory
    agents_info: AgentsInfo
    retrieval_decision: bool
    agents_queries: dict
    response: str


class RetrievalState(TypedDict, total=False):
    query: str
    retrieved_chunks: Dict[Any, Any]
    retrieved_context: Dict[Any, Any]
    summary: str
