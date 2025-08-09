# importing necessary libraries
import re
import requests
import ollama

# ---- Defining Tools ---- #
# 1. Simple calculator tool
def calculator_tool(query):
    """A simple calculator tool."""
    try:
        # Keep only numbers, operators, parentheses, and dots
        cleaned_expr = query.split(' = ')[0]  # Remove anything after '='
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

# ---- Defining Short Term memory ---- #
memory = [
    {
        'role': 'system',
        'content': """
                    You are a helpful AI assistant. You can either perform calculations or search the web.
                    - CALCULATE: <math expression>
                    - SEARCH: <search query>
                    - ANSWER: <your answer>

                    Always use the information from previous conversation if relevant.
                    Output ONLY one of the above actions on the FIRST line.
                    Do not explain your reasoning. Do not add extra commentary.

                    Output ONLY one of the above actions on the FIRST line. 
                    Do not explain your reasoning. Do not add extra commentary.
                    
                    Example:
                    Q: What is 12*(5+2)?
                    A: CALCULATE: 12*(5+2)
                    
                    Q: Who is the president of the USA?
                    A: SEARCH: Who is the president of the USA?
                    
                    Q: Tell me a joke about Cats.
                    A: ANSWER: Sure, here is a joke...
                    """
    }
]

# ---- Defining Agent ---- #
def agent(query, model="llama3.2:latest"):
    # Adding user query to memory
    memory.append({"role": "user", "content": query})

    # Ask LLM with conversation history
    decision_response = ollama.chat(
        model=model,
        messages=memory
    )

    # Extracting the tools name from decision response
    decision = decision_response['message']['content'].strip()

    # save model's reasoning memory
    memory.append({"role": "assistant", "content": decision})

    # Get  first line of (action)
    first_line = decision.split('\n')[0].strip()

    # Exectuing the tools
    if "CALCULATE" in first_line:
        expression = first_line.split("CALCULATE:", 1)[1].strip()
        result = calculator_tool(expression)
    elif "SEARCH:" in first_line:
        search_query = first_line.split("SEARCH:")[1].strip()
        result = web_search(search_query)
    elif "ANSWER:" in first_line:
        result = first_line.split("ANSWER:")[1].strip()
    else:
        result = "Invalid action specified."
    
    # Savinig Result to memory
    memory.append({"role": "assistant", "content": result})

    return result


# --- Example usage --- #
print(agent("Who is the ceo of Google?"))
print(agent("How old is he?"))

# with open("memory.txt", "w") as f:
#     f.write(str(memory))