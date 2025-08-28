import os
import json

from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import Runnable
from langchain_google_genai import ChatGoogleGenerativeAI

from chat_agh.prompts import GENERATION_PROMPT_TEMPLATE

DEFAULT_GENERATION_MODEL = "gemini-2.5-flash"


class GenerationAgent:
    def __init__(self):
        super().__init__()
        self.api_keys = json.loads(os.getenv("GEMINI_API_KEYS", "[]"))
        self.llm = ChatGoogleGenerativeAI(model=DEFAULT_GENERATION_MODEL, api_key=self.api_keys[0])

        self.prompt = PromptTemplate(
            input_variables=["agents_info", "chat_history"],
            template=GENERATION_PROMPT_TEMPLATE
        )
        self.chain: Runnable = self.prompt | self.llm

    def stream(self, chat_history, context):
        start_state = {
            "context": context,
            "chat_history": chat_history
        }
        for chunk in self.chain.stream(start_state):
            yield chunk