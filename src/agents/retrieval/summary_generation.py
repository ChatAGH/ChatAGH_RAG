import os
import json
import random

from pydantic import BaseModel
from langchain_core.runnables import Runnable
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import PydanticOutputParser

from src.prompts import SUMMARY_GENERATION_PROMPT_TEMPLATE
from src.states import RetrievalState

DEFAULT_SUMMARY_GENERATION_MODEL = "gemini-2.5-flash"


class SummaryGenerationOutput(BaseModel):
    summary: str


class SummaryGeneration:
    def __init__(self, model_name: str = DEFAULT_SUMMARY_GENERATION_MODEL):
        self.api_keys = json.loads(os.getenv("GEMINI_API_KEYS", "[]"))
        self.llm = ChatGoogleGenerativeAI(
            model=model_name,
            api_key=random.choice(self.api_keys[0])
        )
        self.prompt = PromptTemplate(
            input_variables=["context"],
            template=SUMMARY_GENERATION_PROMPT_TEMPLATE
        )
        self.output_parser = PydanticOutputParser(pydantic_object=SummaryGenerationOutput)
        self.chain: Runnable = self.prompt | self.llm | self.output_parser

    def __call__(self, state: RetrievalState):
        context = ""
        for retrieved_context in state["retrieved_context"]:
            context += (
                "## CONTEXT"
                f"### Source URL: {retrieved_context.source_url}\n"
                f"### Retrieved chunks: {retrieved_context.chunks}\n"
                f"### Related context: {retrieved_context.related_chunks}\n"
            )

        summary = self.chain.invoke({"context"})
        return {"summary": summary}
