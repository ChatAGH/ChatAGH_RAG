import os
import json
import random

from langchain_core.runnables import Runnable
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

from chat_agh.utils.utils import retry_on_exception, log_execution_time
from chat_agh.prompts import SUMMARY_GENERATION_PROMPT_TEMPLATE
from chat_agh.states import RetrievalState

DEFAULT_SUMMARY_GENERATION_MODEL = "gemini-2.5-flash"


class SummaryGeneration:
    def __init__(self, model_name: str = DEFAULT_SUMMARY_GENERATION_MODEL):
        self.api_keys = json.loads(os.getenv("GEMINI_API_KEYS", "[]"))
        self.llm = ChatGoogleGenerativeAI(
            model=model_name,
            api_key=random.choice(self.api_keys)
        )
        self.prompt = PromptTemplate(
            input_variables=["context", "query"],
            template=SUMMARY_GENERATION_PROMPT_TEMPLATE
        )
        self.chain: Runnable = self.prompt | self.llm

    @log_execution_time
    @retry_on_exception(attempts=3, delay=1, backoff=3)
    def __call__(self, state: RetrievalState):
        context = ""
        for retrieved_context in state["retrieved_context"]:
            context += (
                "## CONTEXT"
                f"### Source URL: {retrieved_context.source_url}\n"
                f"### Retrieved chunks: {retrieved_context.chunks}\n"
                f"### Related context: {retrieved_context.related_chunks}\n"
            )

        summary = self.chain.invoke({"context": context, "query": state["query"]})
        return {"summary": summary}
