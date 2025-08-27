import os
import json
import random
from abc import abstractmethod

from langchain_google_genai import ChatGoogleGenerativeAI

DEFAULT_MODEL = "gemini-2.5-flash"


class BaseAgent:
    def __init__(self, model_name: str = DEFAULT_MODEL, **kwargs):
        self.model_name: str = model_name
        self.api_keys = json.loads(os.getenv("GEMINI_API_KEYS", "[]"))
        self.llm: ChatGoogleGenerativeAI = ChatGoogleGenerativeAI(model=model_name, api_key=self.api_keys[0])

    @abstractmethod
    def _inference(self, *args, **kwargs):
        pass

    def invoke(self, *args, **kwargs):
        self._api_key_rotation()
        return self._inference(*args, **kwargs)

    def _api_key_rotation(self):
        api_key = random.choice(self.api_keys)
        self.llm = ChatGoogleGenerativeAI(model=self.model_name, api_key=api_key)
