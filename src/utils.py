from dataclasses import dataclass

from langchain.schema import BaseMessage


class ChatHistory:
    def __init__(self, messages: list[BaseMessage]):
        self.messages = self._validate_messages(messages)

    def _validate_messages(self, messages: list[BaseMessage]) -> list[BaseMessage]:
        if any(not isinstance(m, BaseMessage) for m in messages):
            raise TypeError("Cannot initialize ChatHistory object, all messages must be of type BaseMessage")
        return messages

    def __getitem__(self, item):
        if isinstance(item, slice):
            return ChatHistory(self.messages[item])
        else:
            return self.messages[item]

    def __str__(self):
        return "\n".join(
            f"  {msg.type.upper()} MESSAGE: {msg.content}" for msg in self.messages
        )


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

