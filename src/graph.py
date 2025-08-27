from langgraph.graph.state import StateGraph, END, START

from src.state import ChatState
from src.nodes import RetrievalNode, OrchestrationNode


class ChatGraph:
    def __init__(self):
        self.workflow = (
            StateGraph(ChatState)
            .add_node("orchestration_node", OrchestrationNode())
            .add_node("retrieval_node", RetrievalNode())
            .add_edge(START, "orchestration_node")
            .add_conditional_edges(
                "orchestration_node",
                lambda state: "retrieval_node" if state["retrieval_decision"] else END
            )
            .add_edge("retrieval_node", "orchestration_node")
            .compile()
        )

    def query(self, question: str) -> str:
        pass

    def invoke(self, chat_history: list):
        pass

    def stream(self, chat_history: list):
        pass
