from langchain_core.messages import HumanMessage
from langgraph.graph.state import StateGraph, END, START

from chat_agh.states import ChatState
from chat_agh.nodes import RetrievalNode, SupervisorNode, GenerationNode, InitialRetrievalNode
from chat_agh.utils.agents_info import AgentsInfo, AgentDetails, RETRIEVAL_AGENTS
from chat_agh.utils.chat_history import ChatHistory


class ChatGraph:
    def __init__(self):
        self.graph = (
            StateGraph(ChatState)
            .add_node("initial_retrieval_node", InitialRetrievalNode(["chunks"]))
            .add_node("supervisor_node", SupervisorNode())
            .add_node("retrieval_node", RetrievalNode())
            .add_node("generation_node", GenerationNode())
            .add_edge(START, "initial_retrieval_node")
            .add_edge("initial_retrieval_node", "supervisor_node")
            .add_conditional_edges(
                "supervisor_node",
                lambda state: "retrieval_node" if state["retrieval_decision"] else END
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
    from chat_agh.utils.utils import logger
    chat_graph = ChatGraph()

    chat_history = ChatHistory(
        messages=[
            HumanMessage("Hej, ile godzin analizy na I semestrze informatyki i systemow inteligentnych?")
        ]
    )
    logger.info("START")
    for c in chat_graph.stream(chat_history):
        print(c)
