import streamlit as st
import random
import time
import os
import json

from crewai import LLM, Agent, Task, Crew, Process

from Agents import CodingAgent, RetrieveHRPolicy
from memory import SQLiteMemory, convert_tuple_list_to_text

# --- Configuration --- #
MODEL_NAME = "ollama/openhermes:latest"
db_path = "supervisor_memory.db"

# Defining LLM
llm = LLM(
    model= MODEL_NAME,
    temperature= 0.7
)

# Instantiate the tools
coding_tool = CodingAgent()
retrival_tool = RetrieveHRPolicy()

# --- Defining All Agents --- #
# Define Coding Agent
coding_agent = Agent(
    role="coding_agent",
    goal="""Write python code based on user queries and answer executing code.""",
    backstory= """You are a helpful assistant. You can write and execute Python code based on user queries and return the output,
      executing the code.""",
    verbose=False,
    tools=[coding_tool],
    llm=llm
)

# define Agent first
rag_agent = Agent(
    role="RAG Agent",
    goal="""Answering user queries from a vector database. You must use the retrieval tool for every query. 
        If the tool's response is 'No relevant documents found.', you must respond with "I don't know the answer."
        Otherwise, use the retrieved documents to formulate your answer. Do not use any other tools or methods.""",
    backstory= """You are a helpful assistant. You can answer user queries from a vector database created from HR policies.""",
    verbose=False,
    tools=[retrival_tool],
    llm=llm
)

# Defining Supervisor Agent
supervisor_agent = Agent(
    role="Supervisor Agent",
    goal="""Analyze user queries and decide whether to route the request to the Coding Agent or the Helper Agent. Be very strict in your decision-making.""",
    backstory="""You are a top-tier assistant that acts as a router for a team of specialized agents.
        Your primary function is to classify user requests. You MUST ALWAYS route the query to one of the following agents, by name:
        - Coding Agent: For tasks that explicitly require writing, running, or debugging code (e.g., "write a Python script", "calculate X using code", "debug this function").
        - RAG Agent: For HR policies question (e.g., "What are employee benifits?", "What are the leave policies?").
        You CAN only select one of these agents.""",
    verbose=False,
    llm=llm
)

# --- Defining Task --- #
# Defining Coading Task
coding_task = Task(
    description=f"""Based on historical context see if the query is related with privious question or not.
    Based on your understanding write clean python code based on the user query and historical context.
        Historical Context: {{context}}
        User Query: {{query}}""",
    expected_output="""Return a json object with keys "code" and "output".""",
    agent=coding_agent
)

# Defining RAG Task
rag_task = Task(
    description=f"""Following is a user Question. First check if the question is related to previously asked question or not.
     Based on your understanding use the retrieval tool to find the relevant answer based on HR policies and the historical context.
        Historical Context: {{context}}
        User Query: {{query}}""",
    expected_output="Return short and to the point answer.",
    agent=rag_agent
)

supervisor_task = Task(
    description=f"""Analyze the user's query and decide on the best course of action.
        - If the query is related to math, computer science or coding or if answering the query requirs generating code, delegate to coding_task.
        - If the query is Human Resources related question, delegate to rag_task.
        - If the query about the conversation history then answer from context and do not suggest any tool.
        Context: {{context}}
        User Query: {{query}}""",
    expected_output="Answer could be 'coding_task' or 'rag_task'. Else The answer will be in Natural language.",
    agent=supervisor_agent,
    possible_tasks=[coding_task, rag_task]  
)

# --- Defining Crew --- #
# Define Crew
coding_crew = Crew(
    agents=[coding_agent],
    tasks=[coding_task],
    verbose=False,
    process=Process.sequential, # The supervisor task will run first, and its output will feed into the next task.
)

rag_crew = Crew(
    agents=[rag_agent],
    tasks=[rag_task],
    verbose=False
)

supervisor_crew = Crew(
    agents= [supervisor_agent],
    tasks= [supervisor_task],
    verbose= False,
    process= Process.sequential
)

### ------ Creating Checkpoint ------ ###
# Save a checkpoint after a key decision is made
memory = SQLiteMemory(db_path=db_path)

### ------ Creating Frontend ------ ###
# Streamed response emulator
def response_generator(result:str):
    for word in result.split():
        yield word + " "
        time.sleep(0.05)

# Function to start a new chat
def new_chat():
    st.session_state.messages = []  # Clear chat history
    checkpoint_id = memory.save_checkpoint("New Session Started", "")
    print(f"\tCheckpoint saved with ID: {checkpoint_id}")

def convert_text_to_json(text_data):
  """
  Converts a given text string to a JSON object.

  Args:
    text_data: A string containing the JSON data.

  Returns:
    A JSON object (dictionary or list) if the conversion is successful,
    otherwise None.
  """
  try:
    # The json.loads() method parses a JSON string and returns a Python dictionary or list.
    json_object = json.loads(text_data)
    return json_object
  except json.JSONDecodeError as e:
    # Handle the case where the input string is not valid JSON
    print(f"Error decoding JSON: {e}")
    return None

# Sidebar with New Chat button
with st.sidebar:
    if st.button("+ New Chat"):
        new_chat()
        st.experimental_rerun()  # Refresh the app

# Title of the Chatbot
st.title("Simple chat")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("How can I help?"):

    # Loading last 3 conversion
    history = memory.load_recent(3)

    # Get the final formatted text
    history = convert_tuple_list_to_text(history)

    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    decision = supervisor_crew.kickoff(inputs={'query': prompt, 'context': history})
    print(f"Decision: {decision}")

    # based on result selecting Agent
    if decision.raw == 'coding_task':
        result = coding_crew.kickoff(inputs= {'query': prompt, 'context': history})
        print(result)   
        # print(f"CODE:\n{result['code']}")
        # print(f"OUTPUT:\n{result['output']}")
        with st.chat_message("assistant"):
            json_result = convert_text_to_json(result.raw)
            response = f"CODE:\n\n```\n{json_result.get('code', "")}\n```\n\nOutput:\n```\n{json_result.get('output', "")}\n```"
            print(response)
            market_placehlder = st.empty()
            output1 = market_placehlder.markdown(f"{response}\n")
            
    elif decision.raw == 'rag_task':
        result = rag_crew.kickoff(inputs= {'query': prompt, 'context': history})
        print(result)
        with st.chat_message("assistant"):
            response = st.write_stream(response_generator(result.raw))
    else:
        result = decision
        print(result)
        with st.chat_message("assistant"):
            response = st.write_stream(response_generator(result.raw))
        print(f"🤖: {decision}")

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})

    # Saving History
    memory.save("User", prompt)
    memory.save("Agent", result.raw)


# # if os.path.exists(db_path):
# #     os.remove(db_path)

# # Defining Control Flow.
# def main():
#     """Main function to kickoff the Crew."""

#     memory = SQLiteMemory(db_path=db_path)
        
#     print("\n\t--- Starting Crew for RAG Agent ---")

#     while True:
#         # Load from the previous checkpoint
#         history = memory.load_recent(3)
#         # Get the final formatted text
#         history = convert_tuple_list_to_text(history)

#         user_query = input("Enter your query (or type 'exit' to quit): ")
#         print(f"\n🙎‍♂️: {user_query}")

#         # Breaking the loop if exit or quit was input
#         if user_query.lower() in ['exit', 'quit']:
#             # Save a checkpoint after a key decision is made
#             checkpoint_id = memory.save_checkpoint("supervisor_agent", decision.raw)
#             print(f"\tCheckpoint saved with ID: {checkpoint_id}")
#             print("Saving History ...")
#             memory.close()
#             break

#         decision = supervisor_crew.kickoff(inputs={'query': user_query, 'context': history})
#         print(f"\t I need to check with {decision}")

#         if decision.raw == 'coding_task':
#             result = coding_crew.kickoff(inputs= {'query': user_query, 'context': history})
            
#             print(f"🤖: {result}")
#         elif decision.raw == 'rag_task':
#             result = rag_crew.kickoff(inputs= {'query': user_query, 'context': history})
#             print(f"🤖: {result}")
#         else:
#             result = decision
#             print(f"🤖: {decision}")

#         # Saving History
#         memory.save("User", user_query)
#         memory.save("Agent", result.raw)


# if __name__ == "__main__":
#     main()


# print("\n\t--- Starting Crew for RAG Agent ---")

    
# # Load from the previous checkpoint
# user_query = input("Enter your query (or type 'exit' to quit): ")
# print(f"\n🙎‍♂️: {user_query}")

# decision = supervisor_crew.kickoff(inputs={'query': user_query, 'context': ""})
# print(f"\t I need to check with {decision}")

# if decision.raw == 'coding_task':
#     result = coding_crew.kickoff(inputs= {'query': user_query, 'context': ""})
#     print(f"🤖: {result}")
# elif decision.raw == 'rag_task':
#     result = rag_crew.kickoff(inputs= {'query': user_query})
#     print(f"🤖: {result}")
# else:
#     result = decision
#     print(f"🤖: {decision}")


# import json

# def convert_text_to_json(text_data):
#   """
#   Converts a given text string to a JSON object.

#   Args:
#     text_data: A string containing the JSON data.

#   Returns:
#     A JSON object (dictionary or list) if the conversion is successful,
#     otherwise None.
#   """
#   try:
#     # The json.loads() method parses a JSON string and returns a Python dictionary or list.
#     json_object = json.loads(text_data)
#     return json_object
#   except json.JSONDecodeError as e:
#     # Handle the case where the input string is not valid JSON
#     print(f"Error decoding JSON: {e}")
#     return None

# # The given text string
# given_text = '{\n  "code": "import random\\nprint(random.randint(0, 10))",\n  "output": 9\n}'

# # Convert the text to a JSON object
# json_result = convert_text_to_json(result.raw)

# print(json_result['code'])
