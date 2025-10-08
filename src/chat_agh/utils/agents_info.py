from dataclasses import dataclass


@dataclass
class AgentDetails:
    name: str
    description: str
    cached_history: dict[str, str] | None


class AgentsInfo:
    def __init__(self, agents_details: list[AgentDetails]):
        self.agents_details = agents_details

    def __str__(self) -> str:
        text = ""
        for detail in self.agents_details:
            text += (
                f"  Agent name: {detail.name}\n"
                f"  Agent description: {detail.description}\n"
            )
            if detail.cached_history is not None:
                cached_history_text = (
                    f"\n    Query: {detail.cached_history['query']}"
                    f"\n    Response: {detail.cached_history['response']}"
                )
                text += f"  Cached Conversation: {cached_history_text}"

        return text


@dataclass
class RetrievalAgentInfo:
    name: str
    vector_store_index_name: str
    description: str


RETRIEVAL_AGENTS = [
    RetrievalAgentInfo(
        name="recrutation_agent",
        vector_store_index_name="rekrutacja",
        description="""
        Agent wyszukujący informacji dotyczących rekrutacji na AGH na różne typy studiów.
        Ma dostęp do regulaminów, kalendarzy, aktualnych informacji i wszystkiego co związane z rekrutacją.
        """,
    ),
    RetrievalAgentInfo(
        name="campus_agent",
        vector_store_index_name="miasteczko",
        description="""
        Agent wyszukujący informacji dotyczących kampusu AGH.
        Ma dostęp do danych o domach studenckich (akademikach), ich przyznawaniu, obowiązujących zasad i regulaminów
        i wszystkich informacji związanych z życiem na kampusie AGH
        """,
    ),
    RetrievalAgentInfo(
        name="dss_agent",
        vector_store_index_name="dss",
        description="""
        Agent wyszukujący informacji z działu spraw studenckich AGH.
        Informacje o procedurach administracyjnych, takich jak składanie wniosków o stypendia,
         urlopy dziekańskie czy wydawanie zaświadczeń.
          Dodatkowo często dostępne są wzory dokumentów, regulaminy studiów i informacje dotyczące akademików, pomocy materialnej czy szeroko pojętej działąlności uczelni.
        """,
    ),
]
