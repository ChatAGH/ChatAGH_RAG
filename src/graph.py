from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph.state import StateGraph, END, START

from src.states import ChatState
from src.nodes import RetrievalNode, SupervisorNode, RETRIEVAL_AGENTS
from src.utils.agents_info import AgentsInfo, AgentDetails
from src.utils.chat_history import ChatHistory


class ChatGraph:
    def __init__(self):
        self.graph = (
            StateGraph(ChatState)
            .add_node("supervisor_node", SupervisorNode())
            .add_node("retrieval_node", RetrievalNode())
            .add_edge(START, "supervisor_node")
            .add_conditional_edges(
                "supervisor_node",
                lambda state: "retrieval_node" if state["retrieval_decision"] else END
            )
            .add_edge("retrieval_node", "supervisor_node")
            .compile()
        )

    def query(self, question: str, **kwargs) -> str:
        chat_history = ChatHistory(messages=[HumanMessage(question)])
        state = ChatState(
            chat_history=chat_history,
            agents_info=AgentsInfo(agents_details=[
                AgentDetails(
                    name=agents_details.name,
                    description=agents_details.description,
                    cached_history=None
                ) for agents_details in RETRIEVAL_AGENTS
            ]),
        )
        return self.graph.invoke(state)["response"]

    def invoke(self, chat_history: list[BaseMessage]):
        pass

    def stream(self, chat_history: list[BaseMessage]):
        pass


if __name__ == "__main__":
    chat_graph = ChatGraph()
    print(chat_graph.query("Jak zostać studentem AGH?"))
