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
        name="main_agent",
        vector_store_index_name="cluster_0",
        description="""
        Agent pobierający informację z głównej strony agh, na której znajdują się podstawowe informacje o uczelni i
        przekierowania do innych domen/informacji. Zawiera również informacje z Centurm Obsługi Kształcenia
        gdzie można znaleźć informacje dotyczące organizacji i koordynacji procesu kształcenia na studiach wyższych i podyplomowych,
        w tym obsługi systemu USOS, zasad i procedur dydaktycznych oraz wsparcia technicznego dla jednostek uczelni.
        
        Agent obsługuje następujące domeny:
        - agh.edu.pl
        - cok.agh.edu.pl
        """,
    ),
    RetrievalAgentInfo(
        name="wh_agent",
        vector_store_index_name="cluster_1",
        description="""
        Agent Wydziału Humanistycznego AGH
        
        Agent obsługuje następujące domeny:
        - wh.agh.edu.pl
        - power.3.5.wh.agh.edu.pl
        - konfpau.wh.agh.edu.pl
        - multisite.wh.agh.edu.pl
        """,
    ),
    # RetrievalAgentInfo(
    #     name="dss_agent",
    #     vector_store_index_name="dss",
    #     description="""
    #     Agent wyszukujący informacji z działu spraw studenckich AGH.
    #     Informacje o procedurach administracyjnych, takich jak składanie wniosków o stypendia,
    #      urlopy dziekańskie czy wydawanie zaświadczeń.
    #       Dodatkowo często dostępne są wzory dokumentów, regulaminy studiów i informacje dotyczące akademików, pomocy materialnej czy szeroko pojętej działąlności uczelni.
    #     """,
    # ),
]
