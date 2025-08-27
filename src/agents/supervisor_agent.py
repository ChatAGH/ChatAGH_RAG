import os
import json
from typing import Optional

from pydantic import BaseModel, model_validator
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import Runnable
from langchain_google_genai import ChatGoogleGenerativeAI

from src.prompts import SUPERVISOR_AGENT_PROMPT_TEMPLATE
from src.utils.agents_info import AgentDetails, AgentsInfo
from src.utils.chat_history import ChatHistory

AGENTS_NAMES = ["recrutation_agent", "dormitories_agent"]
DEFAULT_SUPERVISOR_MODEL = "gemini-2.5-flash"


class SupervisorOutput(BaseModel):
    retrieval_decision: bool
    message: Optional[str] = None
    queries: Optional[dict[str, str]] = None

    @model_validator(mode="before")
    def check_fields_based_on_decision(cls, values):
        decision = values.get('retrieval_decision')
        message = values.get('message')
        queries = values.get('queries')

        if decision:
            if not queries:
                raise ValueError("When retrieval_decision is True, 'queries' must be provided")
            if message is not None:
                raise ValueError("'message' should not be provided when retrieval_decision is True")
        else:
            if not message:
                raise ValueError("When retrieval_decision is False, 'message' must be provided")
            if queries is not None:
                raise ValueError("'queries' should not be provided when retrieval_decision is False")

        if queries:
            for agent in queries.keys():
                if agent not in AGENTS_NAMES:
                    raise ValueError(f"Agent '{agent}' is not defined")

        return values


class SupervisorAgent:
    def __init__(self):
        super().__init__()
        self.api_keys = json.loads(os.getenv("GEMINI_API_KEYS", "[]"))
        self.llm = ChatGoogleGenerativeAI(model=DEFAULT_SUPERVISOR_MODEL, api_key=self.api_keys[0])

        self.output_parser = PydanticOutputParser(pydantic_object=SupervisorOutput)
        self.prompt = PromptTemplate(
            input_variables=["agents_info", "chat_history", "latest_user_message"],
            template=SUPERVISOR_AGENT_PROMPT_TEMPLATE
        )
        self.chain: Runnable = self.prompt | self.llm | self.output_parser

    def invoke(self, agents_info: AgentsInfo, chat_history: ChatHistory):
        return self.chain.invoke({
            "agents_info": agents_info,
            "chat_history": chat_history[:-1],
            "latest_user_message": chat_history[-1].content
        })


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv("/Users/wnowogorski/PycharmProjects/ChatAGH_RAG/.env")

    from langchain.schema import HumanMessage, AIMessage

    from src.utils.chat_history import ChatHistory

    agent = SupervisorAgent()
    res = agent.invoke(
        agents_info=AgentsInfo([
            AgentDetails(
                name="recrutation_agent",
                description="Agent retrieving informations about recrutation",
                cached_history={
                    "query": "Jak zostać studentem AGH?",
                    "response": "Musisz przejsc proces rekrutacji"
                }
            )
        ]),
        chat_history=ChatHistory(
            messages=[
                HumanMessage("Hej"),
                AIMessage("Cześć!, Jak mogę ci pomóc?"),
                HumanMessage("Jak dostać się na AGH?")
            ]
        ),
    )
    print(res)