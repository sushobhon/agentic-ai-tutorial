# Agentic AI Tutorial (From Basic)

For Learning perpose we will only use open source tools, like `Ollama` as LLM, `wikipedia` for web search.

### Lession 1: Creating simple Agents
- Created a simple Agent which desides Which tools to use between 2 tools. i. `CALCULATOR`, ii. `RAG AGENT`.
- Tools are basically 2 function.
- Agents will deside which tools to use.

### Lession 2: Added Short Term Memory
- Added a short term memory to agents.
- Used a list of message too store short term message.

### Lession 3: Added Long term Memory
- Added Long term memory by using SQL Light Data Base.
- Added Loop in the Agents.

### Lession 4: Added Supervisor Agent to supervise the Agent
- Added a supervisor Agent to supervise the questions.
- Supervisor Agent will deside which Agent to call.

### Lession 5: Added an Simple UI for Chat
- A simple UI using `Streamlit` to chat.

# How to run?
### Create Environment
Create a new vertual environments using `pyproject.toml` file.
```
uv venv .venv
```
Activate the environment
```
.venv\Scripts\activate
```
Now, install all the packages.
