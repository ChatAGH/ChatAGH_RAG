from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph.state import StateGraph, END, START

from src.state import ChatState
from src.nodes import RetrievalNode, OrchestrationNode
from src.utils.agents_info import AgentsInfo, AgentDetails
from src.utils.chat_history import ChatHistory


class ChatGraph:
    def __init__(self):
        self.builder = (
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

    def query(self, question: str, **kwargs) -> str:
        chat_history = ChatHistory(messages=[HumanMessage(question)])
        agents_info = kwargs["agents_info"]
        state = ChatState(
            chat_history=chat_history,
            agents_info=agents_info,
        )
        return self.builder.invoke(state)["response"]

    def invoke(self, chat_history: list[BaseMessage]):
        pass

    def stream(self, chat_history: list[BaseMessage]):
        pass


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv("/Users/wnowogorski/PycharmProjects/ChatAGH_RAG/.env")

    chat_graph = ChatGraph()
    print(chat_graph.query(
        "Jak zostać studentem AGH?", agents_info=[
            AgentDetails(
                name="recrutation_agent",
                description="Agent retrieving informations about rectutation",
                cached_history=None
            )
        ]
    ))
