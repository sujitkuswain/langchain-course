from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langsmith import traceable

load_dotenv()

MAX_ITERATIONS = 10
MODEL="gpt-5-nano"

# Tools for agent

@tool
def get_product_price(product: str) -> float:
    """
    Lookup the price of product in catalog.
    """
    print(f" >> Executing get_product_price with (product: {product})")
    prices={"headphones": 49.30, "laptop": 1000.99, "monitor": 379.80}
    return prices.get(product, 0)

@tool
def apply_discount(price: float, price_tier: str) -> float:
    """Apply a discount tier to a product and return the final price. Available tiers are "bronze", "silver", and "gold".
    """
    print(f" >> Executing apply_discount with (price: {price}, price_tier: {price_tier})")
    discount_percentages = {"bronze": 5, "silver": 10, "gold": 20}
    discount = discount_percentages.get(price_tier, 0)
    return round(price * (1 - discount / 100), 2)

# agent loop

@traceable(name="Langchain Agent Loop")
def run_agent(question: str):
    pass


if __name__ == "__main__":
    print("Hello langchain agent (.bind_tools)!")
    print()
    result = "What is the price of laptop after applying a gold discount?"