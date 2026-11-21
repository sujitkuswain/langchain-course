from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from tavily import TavilyClient

tavily = TavilyClient()

load_dotenv()


@tool
def search(query: str) -> str:
    """
    Tool that search over internet

    Args:
        query: The query to search for

    Returns:
        The search results
    """
    print(f"Searching for {query}")
    return tavily.search(query)


llm = ChatOpenAI(model="gpt-5-nano")
tools = [search]
agent = create_agent(model=llm, tools=tools)


def main():
    result = agent.invoke(
        {"messages": HumanMessage(content="How is the weather in Tokyo?")}
    )
    print(result)


if __name__ == "__main__":
    main()
