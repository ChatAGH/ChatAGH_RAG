from typing import Any, Dict

from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import Runnable
from langchain_google_genai import ChatGoogleGenerativeAI

from chat_agh.prompts import SUMMARY_GENERATION_PROMPT_TEMPLATE
from chat_agh.states import RetrievalState
from chat_agh.utils import (
    RetrievedContext,
    log_execution_time,
    retry_on_exception,
)
from chat_agh.utils.utils import GEMINI_API_KEY

DEFAULT_SUMMARY_GENERATION_MODEL = "gemini-2.5-flash"


class SummaryGeneration:
    def __init__(self, model_name: str = DEFAULT_SUMMARY_GENERATION_MODEL) -> None:
        self.llm = ChatGoogleGenerativeAI(model=model_name, api_key=GEMINI_API_KEY)
        self.prompt = PromptTemplate(
            input_variables=["context", "query"],
            template=SUMMARY_GENERATION_PROMPT_TEMPLATE,
        )
        self.chain: Runnable[Dict[str, Any], Any] = self.prompt | self.llm

    @log_execution_time
    @retry_on_exception(attempts=3, delay=1, backoff=3)
    def __call__(self, state: RetrievalState) -> Dict[str, Any]:
        context = ""
        retrieved_contexts: list[RetrievedContext] = state["retrieved_context"]
        for retrieved_context in retrieved_contexts:
            context += "## CONTEXT\n" + retrieved_context.text + "\n"

        summary = self.chain.invoke({"context": context, "query": state["query"]})
        return {"summary": summary}
