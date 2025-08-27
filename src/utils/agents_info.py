from dataclasses import dataclass


@dataclass
class AgentDetails:
    name: str
    description: str
    cached_history: dict[str, str] | None


class AgentsInfo:
    def __init__(self, agents_details: list[AgentDetails]):
        self.agents_details = agents_details

    def __str__(self):
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
