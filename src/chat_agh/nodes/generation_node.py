from typing import List

from langgraph.config import get_stream_writer
from langchain_core.documents import Document

from chat_agh.states import ChatState
from chat_agh.agents import GenerationAgent
from chat_agh.utils.utils import log_execution_time, retry_on_exception


class GenerationNode:
    def __init__(self):
        self.agent = GenerationAgent()

    @retry_on_exception(attempts=2, delay=1, backoff=3)
    @log_execution_time
    def __call__(self, state: ChatState) -> dict:
        writer = get_stream_writer()

        if any(agent.cached_history for agent in state["agents_info"].agents_details):
            context = str(state["agents_info"])
        else:
            documents: List[Document] = state["context"]
            context = "\n".join([document.page_content for document in documents])

        args ={
            "context": context,
            "chat_history": state["chat_history"]
        }
        response = ""
        for response_chunk in self.agent.stream(**args):
            writer(response_chunk)
            response += response_chunk.content

        return {"response": response}