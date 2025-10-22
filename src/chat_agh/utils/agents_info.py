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

        Obsługuje domeny:
        - agh.edu.pl
        - cok.agh.edu.pl
        """,
    ),
    RetrievalAgentInfo(
        name="Wydział Humanistyczny agent",
        vector_store_index_name="cluster_1",
        description="""
        Agent Wydziału Humanistycznego AGH

        Obsługuje domeny:
        - wh.agh.edu.pl
        - power.3.5.wh.agh.edu.pl
        - konfpau.wh.agh.edu.pl
        - multisite.wh.agh.edu.pl
        """,
    ),
    RetrievalAgentInfo(
        name="Wydział Informatyczny agent",
        vector_store_index_name="cluster_2",
        description="""
        Agent Wydziału Informatyki AGH

        Obsługuje domeny:
        - informatyka.agh.edu.pl
        """,
    ),
    RetrievalAgentInfo(
        name="Podyplomoce agent",
        vector_store_index_name="cluster_3",
        description="""
        Agent zajmujący sie studiami podyplomowymi agh oraz studenckimi kołami naukowymi na agh.

        Obsługuje domeny:
        - informatyka.podyplomowe.agh.edu.pl
        - podyplomowe.agh.edu.pl
        - skn.agh.edu.pl
        """,
    ),
    RetrievalAgentInfo(
        name="Wydział Inżynierii Lądowej i Gospodarki Zasobami agent",
        vector_store_index_name="cluster_4",
        description="""
        Agent Wydział Inżynierii Lądowej i Gospodarki Zasobami

        Obsługuje domeny:
        - wilgz.agh.edu.pl
        """,
    ),
    RetrievalAgentInfo(
        name="Oferta badawcza agh agent",
        vector_store_index_name="cluster_5",
        description="""
        Infmroacje o ofercie badawczej agh, bazie aparatury, zespołach badawczych itd.

        Obsługuje domeny:
        - oferta-badawcza.agh.edu.pl
        """,
    ),
    RetrievalAgentInfo(
        name="SKOS AGH",
        vector_store_index_name="cluster_6",
        description="""
        System Informacyjny AGH, informacje o pracownikach, ciałach kolegialnych i jednostkach organizacyjnych.

        Obsługuje domeny:
        - skos.agh.edu.pl
        - old.skos.agh.edu.pl
        """,
    ),
    RetrievalAgentInfo(
        name="Sprawy studenckie",
        vector_store_index_name="cluster_7",
        description="""

        """,
    ),
]
