from typing import TypedDict

from src.utils import ChatHistory, AgentsInfo


class ChatState(TypedDict):
    chat_history: ChatHistory
    agents_info: AgentsInfo
    retrieval_decision: bool
    agents_queries: dict
    response: str