from typing import Dict, List, Any, Iterator
from langgraph.config import get_stream_writer
from langchain_core.documents import Document

from chat_agh.states import ChatState
from chat_agh.agents import GenerationAgent
from chat_agh.utils.utils import log_execution_time, retry_on_exception


def _extract_text(payload: Any) -> Any:
    """Extract string content from common payload shapes."""
    if isinstance(payload, str):
        return payload
    if isinstance(payload, dict) and isinstance(payload.get("content"), str):
        return payload["content"]
    content = getattr(payload, "content", None)
    if isinstance(content, str):
        return content
    raise TypeError("GenerationAgent returned payload without string content")


def _build_context(state: ChatState) -> str:
    """Prefer cached history; otherwise join retrieved documents."""
    if any(agent.cached_history for agent in state["agents_info"].agents_details):
        return str(state["agents_info"])
    documents: List[Document] = state["context"]
    return "\n".join(doc.page_content for doc in documents)


class GenerationNode:
    def __init__(self) -> None:
        self.agent = GenerationAgent()

    def invoke(self, state: ChatState) -> Dict[str, str]:
        args = {"context": _build_context(state), "chat_history": state["chat_history"]}
        result = self.agent.invoke(**args)
        return {"response": _extract_text(result)}

    def stream(self, state: ChatState) -> Iterator[Dict[str, str]]:
        args = {"context": _build_context(state), "chat_history": state["chat_history"]}
        for chunk in self.agent.stream(**args):
            yield {"response": _extract_text(chunk)}

    @retry_on_exception(attempts=2, delay=1, backoff=3)
    @log_execution_time
    def __call__(self, state: ChatState) -> Dict[str, str]:
        writer = get_stream_writer()
        args = {"context": _build_context(state), "chat_history": state["chat_history"]}

        if writer is not None and hasattr(self.agent, "stream"):
            final_text = ""
            for chunk in self.agent.stream(**args):
                writer(chunk)
                final_text += _extract_text(chunk)
            return {"response": final_text}

        result = self.agent.invoke(**args)
        return {"response": _extract_text(result)}
