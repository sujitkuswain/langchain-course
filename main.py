from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from typing import List
from pydantic import BaseModel, Field

class Source(BaseModel):
    """
    Schema for source used by agents.
    """
    url: str = Field(description="The URL of the source.")

class AgentResponse(BaseModel):
    """
    Schema for agent response with answers and sources.
    """
    answer: str = Field(description="TThe agent's answer to the user's query.")
    sources: List[Source] = Field(default_factory=list, description="List of sources used to generate the answer.")

load_dotenv()

llm = ChatOpenAI(model="gpt-5-nano")
tools = [TavilySearch()]
agent = create_agent(model=llm, tools=tools, response_format=AgentResponse)


def main():
    result = agent.invoke(
        {"messages": HumanMessage(content="How is the weather in Tokyo?")}
    )
    print(result)


if __name__ == "__main__":
    main()
