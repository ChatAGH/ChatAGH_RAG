from typing import Dict, TypedDict

from langchain_core.documents import Document

from chat_agh.utils.agents_info import AgentsInfo
from chat_agh.utils.chat_history import ChatHistory
from chat_agh.utils.utils import RetrievedContext


class ChatState(TypedDict, total=False):
    context: list[Document]
    chat_history: ChatHistory
    agents_info: AgentsInfo
    retrieval_decision: bool
    agents_queries: dict[str, str]
    response: str


class RetrievalState(TypedDict, total=False):
    query: str
    retrieved_chunks: Dict[str, list[Document]]
    retrieved_context: list[RetrievedContext]
    summary: str
