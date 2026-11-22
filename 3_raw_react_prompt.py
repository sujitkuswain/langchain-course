import inspect
import json

from dotenv import load_dotenv
from langsmith import traceable
from openai import OpenAI
import re

load_dotenv()

MAX_ITERATIONS = 10
MODEL = "gpt-5-nano"
client = OpenAI()


# --- Tools (LangChain @tool decorator) ---


@traceable(run_type="tool")
def get_product_price(product: str) -> float:
    """Look up the price of a product in the catalog."""
    print(f"    >> Executing get_product_price(product='{product}')")
    prices = {"laptop": 1299.99, "headphones": 149.95, "keyboard": 89.50}
    return prices.get(product, 0)


@traceable(run_type="tool")
def apply_discount(price: float, discount_tier: str) -> float:
    """Apply a discount tier to a price and return the final price.
    Available tiers: bronze, silver, gold."""
    print(f"    >> Executing apply_discount(price={price}, discount_tier='{discount_tier}')")
    discount_percentages = {"bronze": 5, "silver": 12, "gold": 23}
    discount = discount_percentages.get(discount_tier, 0)
    return round(price * (1 - discount / 100), 2)

tools={
    "get_product_price": get_product_price,
    "apply_discount": apply_discount,
}

def get_tool_description(tool_dict):
    description = []
    for tool_name, tool_func in tool_dict.items():
        original_func = getattr(tool_func, "__wrapped__", tool_func)
        signature= inspect.signature(original_func)
        docstring = inspect.getdoc(original_func) or ""
        description.append(f"{tool_name}{signature}- {docstring}")
    return "\n".join(description)

tool_descriptions = get_tool_description(tools)
tool_names = ", ".join(tools.keys())

react_prompt = f"""
STRICT RULES — you must follow these exactly:
1. NEVER guess or assume any product price. You MUST call get_product_price first to get the real price.
2. Only call apply_discount AFTER you have received a price from get_product_price. Pass the exact price returned by get_product_price — do NOT pass a made-up number.
3. NEVER calculate discounts yourself using math. Always use the apply_discount tool.
4. If the user does not specify a discount tier, ask them which tier to use — do NOT assume one.

Answer the following questions as best you can. You have access to the following tools:

{tool_descriptions}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action, as comma separated values
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {{question}}
Thought:"""

@traceable(run_type="llm")
def openai_chat_traced(model, messages, stop=None):
    # gpt-5 models reject `stop` and `temperature`, so we cut at the stop words ourselves
    response = client.chat.completions.create(model=model, messages=messages)
    message = response.choices[0].message
    for stop_word in stop or []:
        message.content = (message.content or "").split(stop_word)[0]
    return response

# --- Agent Loop ---
@traceable(name="OpenAI Agent Loop")
def run_agent(question: str):    
    print(f"Question: {question}")
    print("=" * 60)
    
    prompt = react_prompt.format(question=question)
    scratchpad=""

    for iteration in range(1, MAX_ITERATIONS + 1):
        print(f"\n--- Iteration {iteration} ---")
        
        full_prompt = prompt + scratchpad

        # Difference 5: client.chat.completions.create() directly instead of llm_with_tools.invoke()
        response = openai_chat_traced(
            model=MODEL,
            messages=[{"role": "user", "content": full_prompt}],
            stop=["\nObservation"],
            )
        
        
        ai_message = response.choices[0].message

        tool_calls = ai_message.tool_calls

        # If no tool calls, this is the final answer
        if not tool_calls:
            print(f"\nFinal Answer: {ai_message.content}")
            return ai_message.content

        # Process only the FIRST tool call — force one tool per iteration
        tool_call = tool_calls[0]
        # Difference 6: Attribute access (.function.name) instead of dict access (.get("name"))
        tool_name = tool_call.function.name
        tool_args = json.loads(tool_call.function.arguments)  # OpenAI sends a JSON string

        print(f"  [Tool Selected] {tool_name} with args: {tool_args}")

        tool_to_use = tools_dict.get(tool_name)
        if tool_to_use is None:
            raise ValueError(f"Tool '{tool_name}' not found")

        # Difference 7: Direct function call instead of tool.invoke()
        observation = tool_to_use(**tool_args)


        print(f"  [Tool Result] {observation}")

        messages.append(ai_message.model_dump(exclude_none=True))
        messages.append(
            {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": str(observation),
            }
        )

    print("ERROR: Max iterations reached without a final answer")
    return None


if __name__ == "__main__":
    print("Hello LangChain Agent (.bind_tools)!")
    print()
    result = run_agent("What is the price of a laptop after applying a gold discount?")