import json
import os
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, model_validator
from langchain_core.documents import Document
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import Runnable
from langchain_google_genai import ChatGoogleGenerativeAI

from chat_agh.prompts import SUPERVISOR_AGENT_PROMPT_TEMPLATE
from chat_agh.utils.agents_info import AgentDetails, AgentsInfo, RETRIEVAL_AGENTS
from chat_agh.utils.chat_history import ChatHistory


DEFAULT_SUPERVISOR_MODEL = "gemini-2.5-flash"


class SupervisorOutput(BaseModel):
    retrieval_decision: bool
    queries: Optional[dict[str, str]] = None

    @model_validator(mode="before")
    def check_fields_based_on_decision(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        decision = values.get("retrieval_decision")
        queries = values.get("queries")

        if decision:
            if not queries:
                raise ValueError(
                    "When retrieval_decision is True, 'queries' must be provided"
                )
        else:
            if queries is not None:
                raise ValueError(
                    "'queries' should not be provided when retrieval_decision is False"
                )

        agents_names = [agent.name for agent in RETRIEVAL_AGENTS]
        if queries:
            for agent in queries.keys():
                if agent not in agents_names:
                    raise ValueError(f"Agent '{agent}' is not defined")

        return values


class SupervisorAgent:
    def __init__(self) -> None:
        super().__init__()
        self.api_keys = json.loads(os.getenv("GEMINI_API_KEYS", "[]"))
        self.llm = ChatGoogleGenerativeAI(
            model=DEFAULT_SUPERVISOR_MODEL, api_key=self.api_keys[0]
        )

        self.output_parser = PydanticOutputParser(pydantic_object=SupervisorOutput)
        self.prompt = PromptTemplate(
            input_variables=["agents_info", "chat_history", "latest_user_message"],
            template=SUPERVISOR_AGENT_PROMPT_TEMPLATE,
        )
        self.chain: Runnable[Dict[str, Any], SupervisorOutput] = (
            self.prompt | self.llm | self.output_parser
        )

    def invoke(
        self,
        agents_info: AgentsInfo,
        chat_history: ChatHistory,
        context: Optional[List[Document]] = None,
    ) -> SupervisorOutput:
        return self.chain.invoke(
            {
                "context": context,
                "agents_info": agents_info,
                "chat_history": chat_history[:-1],
                "latest_user_message": chat_history[-1].content,
            }
        )


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv("/.env")

    from langchain.schema import HumanMessage, AIMessage

    from chat_agh.utils.chat_history import ChatHistory
    from chat_agh.utils.chat_history import ChatHistory

    agent = SupervisorAgent()
    res = agent.invoke(
        agents_info=AgentsInfo(
            [
                AgentDetails(
                    name="recrutation_agent",
                    description="Agent retrieving informations about recrutation",
                    cached_history={
                        "query": "Jak zostać studentem AGH?",
                        "response": "Musisz przejsc proces rekrutacji",
                    },
                )
            ]
        ),
        chat_history=ChatHistory(
            messages=[
                HumanMessage("Hej"),
                AIMessage("Cześć!, Jak mogę ci pomóc?"),
                HumanMessage("Jak dostać się na AGH?"),
            ]
        ),
    )
    print(res)
