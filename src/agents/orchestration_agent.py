from typing import Optional

from pydantic import BaseModel, model_validator
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import Runnable
from langchain.schema import BaseMessage

from base_agent import BaseAgent
from src.prompts import ORCHESTRATION_AGENT_PROMPT_TEMPLATE

AGENTS_NAMES = ["recrutation_agent", "dormitories_agent"]


class OrchestrationOutput(BaseModel):
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


class OrchestrationAgent(BaseAgent):
    def __init__(self):
        super().__init__()
        self.output_parser = PydanticOutputParser(pydantic_object=OrchestrationOutput)
        self.prompt = PromptTemplate(
            input_variables=["agents_info", "chat_history", "latest_user_message"],
            template=ORCHESTRATION_AGENT_PROMPT_TEMPLATE
        )
        self.chain: Runnable = self.prompt | self.llm | self.output_parser

    def _inference(self, agents_info, chat_history: list[BaseMessage]):
        return self.chain.invoke({
            "agents_info": agents_info,
            "chat_history": chat_history[:-1],
            "latest_user_message": chat_history[-1].content
        })


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv("/Users/wnowogorski/PycharmProjects/ChatAGH_RAG/.env")

    from langchain.schema import HumanMessage, AIMessage

    from src.utils import ChatHistory

    agent = OrchestrationAgent()
    res = agent.invoke(
        agents_info="""
            AGENT_NAME: recrutation_agent
            DESCRIPTION: Agent which retrieves informations about recrutation.
            HISTORY: Query: Ile kosztuja studja stacjonarne na AGH? Answer: Są darmowe.
            
            AGENT NAME: dormitories_agent
            DESCRIPTION: Agent which retrieves informations about accomodation at university.
            History: Query: Ile jest akademików? Answer: 1300
        """,
        chat_history=ChatHistory(
            messages=[
                HumanMessage("Ile jest akademików?")
            ]
        ),
    )
    print(res)