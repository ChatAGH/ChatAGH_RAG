from langgraph.config import get_stream_writer

from src.chat_agh.states import ChatState
from src.chat_agh.agents import GenerationAgent
from src.chat_agh.utils.utils import log_execution_time, retry_on_exception


class GenerationNode:
    def __init__(self):
        self.agent = GenerationAgent()

    @retry_on_exception(attempts=2, delay=1, backoff=3)
    @log_execution_time
    def __call__(self, state: ChatState) -> dict:
        writer = get_stream_writer()
        args ={
            "agents_info": state["agents_info"],
            "chat_history": state["chat_history"]
        }
        response = ""
        for response_chunk in self.agent.stream(**args):
            writer(response_chunk)
            response += response_chunk.content

        return {"response": response}