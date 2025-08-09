import ollama
import re
import time
import requests
from duckduckgo_search import DDGS

# --- Creating tools for the agents --- #
# 1. Simple calculator tool
def calculator_tool(query):
    """A simple calculator tool."""
    try:
        # Keep only numbers, operators, parentheses, and dots
        cleaned_expr = query.split('=')[0]  # Remove anything after '='
        cleaned_expr = "".join(cleaned_expr).strip()
        return str(eval(cleaned_expr))
    except Exception as e:
        return f"Error in calculation: {e}"

# 2. Wikipedia search tool
def web_search(query):
    url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "list": "search",
        "srsearch": query,
        "format": "json"
    }
    try:
        response = requests.get(url, params=params)
        data = response.json()
        if "query" in data and "search" in data["query"]:
            results = [item['snippet'] for item in data['query']['search']]
            return "\n".join(results) if results else "No results found on Wikipedia."
        else:
            return "No results found."
    except Exception as e:
        return f"Error searching Wikipedia: {e}"


# --- Defining Agent --- #
def agent(query, model="llama3.2:latest"):
    # Step 1: Ask the LLM to deside what to do
    system_prompt = """
    You are a helpful AI assistant. You can either perform calculations or search the web.
    - CALCULATE: <math expression>
    - SEARCH: <search query>
    - ANSWER: <your answer>

    Output ONLY one of the above actions on the FIRST line. 
    Do not explain your reasoning. Do not add extra commentary.
    
    Example:
    Q: What is 12*(5+2)?
    A: CALCULATE: 12*(5+2)
    
    Q: Who is the president of the USA?
    A: SEARCH: Who is the president of the USA?
    
    Q: Tell me a joke about Cats.
    A: ANSWER: Sure, here is a joke:
    """
    decision_response = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query}
        ]
    )

    decision = decision_response['message']['content'].strip()

    # Normalize the decision
    first_line = decision.split('\n')[0].strip()

    print(f"\tDecision made by LLM: {first_line}")

    # STEP 2: Execute the decision
    if "CALCULATE" in first_line:
        expression = first_line.split("CALCULATE:", 1)[1].strip()
        return calculator_tool(expression)
    elif "SEARCH:" in first_line:
        query = first_line.split("SEARCH:")[1].strip()
        return web_search(query)
    elif "ANSWER:" in first_line:
        return first_line.split("ANSWER:")[1].strip()
    else:
        return "I didn't understand the instruction. Please try again with a clear command."

# --- Example usage --- #
# 3. Try it
print(agent("What is 15 * (3 + 2)?"))
print(agent("What is the Capital of India?"))
print(agent("Tell me a joke about girls"))