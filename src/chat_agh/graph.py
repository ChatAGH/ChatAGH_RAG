from langchain_core.messages import HumanMessage
from langgraph.graph.state import StateGraph, END, START

from src.chat_agh.states import ChatState
from src.chat_agh.nodes import RetrievalNode, SupervisorNode, GenerationNode, RETRIEVAL_AGENTS
from src.chat_agh.utils.agents_info import AgentsInfo, AgentDetails
from src.chat_agh.utils.chat_history import ChatHistory


class ChatGraph:
    def __init__(self):
        self.graph = (
            StateGraph(ChatState)
            .add_node("supervisor_node", SupervisorNode())
            .add_node("retrieval_node", RetrievalNode())
            .add_node("generation_node", GenerationNode())
            .add_edge(START, "supervisor_node")
            .add_conditional_edges(
                "supervisor_node",
                lambda state: "retrieval_node" if state["retrieval_decision"] else "generation_node"
            )
            .add_edge("retrieval_node", "generation_node")
            .add_edge("generation_node", END)
            .compile()
        )

    def query(self, question: str, **kwargs) -> str:
        chat_history = ChatHistory(messages=[HumanMessage(question)])
        return self.invoke(chat_history)

    def invoke(self, chat_history: ChatHistory):
        state = ChatState(
            chat_history=chat_history,
            agents_info=self._get_agents_info()
        )
        return self.graph.invoke(state)["response"]

    def stream(self, chat_history: ChatHistory):
        state = ChatState(
            chat_history=chat_history,
            agents_info=self._get_agents_info()
        )
        for response_chunk in self.graph.stream(state, stream_mode="custom"):
            yield response_chunk.content

    def _get_agents_info(self):
        return AgentsInfo(
            agents_details=[
                AgentDetails(
                    name=agents_details.name,
                    description=agents_details.description,
                    cached_history=None
                ) for agents_details in RETRIEVAL_AGENTS
            ]
        )

if __name__ == "__main__":
    chat_graph = ChatGraph()
    # print(chat_graph.query("Jak zostać studentem AGH?"))

    chat_history = ChatHistory(
        messages=[
            HumanMessage("Jak dostać się na AGH?")
        ]
    )
    for c in chat_graph.stream(chat_history):
        print(c)
