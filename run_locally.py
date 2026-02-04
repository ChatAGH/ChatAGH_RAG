from langchain_core.messages import HumanMessage

from dotenv import load_dotenv

from chat_agh.graph import ChatGraph
from chat_agh.utils.chat_history import ChatHistory
from chat_agh.utils.utils import logger

if __name__ == "__main__":
    load_dotenv("/Users/wnowogorski/PycharmProjects/ChatAGH_RAG/.env")

    chat_graph = ChatGraph(enable_retrieval=False)

    chat_history = ChatHistory(
        messages=[HumanMessage("Hejka, kto jest dziekanem WEAIiIB AGH?")]
    )
    logger.info("START")
    res = chat_graph.invoke(chat_history)
    print(res)
