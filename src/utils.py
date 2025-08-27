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
        return "\n\n".join(
            f"{msg.type.upper()} MESSAGE: {msg.content}" for msg in self.messages
        )
