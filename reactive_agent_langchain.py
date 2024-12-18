Developing a **Reactive Agent** using LangChain involves implementing an agent capable of **reacting dynamically to user inputs** based on its tools and capabilities. Reactive agents in LangChain use a **decision-making framework** to interpret inputs and decide what action to take based on the tools available.

---

### **Steps to Build a Reactive Agent Using LangChain**

1. **Choose the LLM**: The agent requires a foundational model (e.g., OpenAI GPT, Hugging Face models, or AWS Bedrock models).

2. **Define Tools**: Tools are the specific capabilities the agent can use (e.g., search engines, calculators, or document retrieval).

3. **Initialize the Agent**: Use LangChain’s `initialize_agent` function to tie the tools and the model together.

4. **Run the Agent**: Feed user queries to the agent, and it will decide the appropriate tool/action to respond.

---

### **Example Code: Reactive Agent**

Below is a practical implementation of a **Reactive Agent** in LangChain:

#### **1. Define the Tools**
Define tools that the agent can use, such as search, calculation, or retrieval.

```python
from langchain.agents import Tool, initialize_agent
from langchain.tools import tool
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
import math

# Tool: Calculator
@tool
def calculate(expression: str) -> str:
    """Evaluate a mathematical expression."""
    try:
        result = eval(expression)
        return f"The result of {expression} is {result}"
    except Exception as e:
        return f"Error in calculation: {e}"

# Tool: Summarizer
@tool
def summarize_text(text: str) -> str:
    """Summarize the given text."""
    llm = OpenAI(model="gpt-4")
    prompt = PromptTemplate(
        input_variables=["text"],
        template="Summarize the following text:\n{text}\n\nSummary:",
    )
    return llm(prompt.format(text=text))
```

#### **2. Create and Configure the Reactive Agent**
Combine tools into an agent and configure it to react dynamically to queries.

```python
# Define the tools
tools = [
    Tool(name="Calculator", func=calculate, description="Performs mathematical calculations."),
    Tool(name="Summarizer", func=summarize_text, description="Summarizes a block of text."),
]

# Initialize the LLM
llm = OpenAI(model="gpt-4")

# Initialize the agent
reactive_agent = initialize_agent(tools, llm, agent="zero-shot-react-description")
```

#### **3. Run the Agent**
Provide user queries and observe how the agent decides which tool to use.

```python
# Test the agent with various queries
queries = [
    "What is the result of 25 * 4 + 10?",
    "Summarize this text: Artificial Intelligence is transforming industries...",
    "Calculate the square root of 144.",
]

for query in queries:
    print(f"Query: {query}")
    response = reactive_agent.run(query)
    print(f"Response: {response}\n")
```

---

### **Key Features of Reactive Agents**

1. **Dynamic Decision Making**: The agent decides at runtime which tool to invoke.
2. **Action Traces**: LangChain provides a step-by-step trace of actions taken by the agent.
3. **Zero-Shot Reasoning**: The agent doesn't need pre-training for a specific workflow—it reacts based on prompt-engineered guidance.

---

### **How It Works Internally**
1. **Agent Prompting**: LangChain uses a **custom prompt** for the agent to understand how to interact with tools.
2. **Tool Invocation**: The agent "decides" which tool to use based on the query context.
3. **Looping (Optional)**: For complex tasks, the agent might use multiple tools sequentially.

---

### **Example Output**

**Query**: "What is the result of 25 * 4 + 10?"  
**Response**: "The result of 25 * 4 + 10 is 110."

**Query**: "Summarize this text: Artificial Intelligence is transforming industries..."  
**Response**: "Artificial Intelligence is driving significant transformation across industries by improving automation, decision-making, and innovation."

**Query**: "Calculate the square root of 144."  
**Response**: "The result of math.sqrt(144) is 12."

---

This implementation demonstrates how a **Reactive Agent** in LangChain can dynamically interpret queries and react by invoking relevant tools, making it ideal for building flexible and intelligent Gen AI systems.
