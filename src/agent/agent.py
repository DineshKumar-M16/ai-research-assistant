from langchain_ollama import ChatOllama
from langchain_core.tools import tool

from langgraph.graph import StateGraph, START, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition

from src.agent.rag_tool import search_documents_tool

# -----------------------------
# 1. Create RAG Search Tool
# -----------------------------

@tool
def research_search(query: str) -> str:
    """
    Search the research documents for relevant information.
    """

    return search_documents_tool(query)

tools = [research_search]

# -----------------------------
# 2. Create Llama model
# -----------------------------

llm = ChatOllama(
    model="llama3.2",
    temperature=0
)

# Give the model access to our tools
llm_with_tools = llm.bind_tools(tools)

# -----------------------------
# 3. Agent Node
# -----------------------------

def agent(state: MessagesState):

    response = llm_with_tools.invoke(
        state["messages"]
    )

    return {
        "messages": [response]
    }

# -----------------------------
# 4. Build LangGraph
# -----------------------------

builder = StateGraph(MessagesState)

builder.add_node(
    "agent",
    agent
)

builder.add_node(
    "tools",
    ToolNode(tools)
)

builder.add_edge(
    START,
    "agent"
)

builder.add_conditional_edges(
    "agent",
    tools_condition
)

builder.add_edge(
    "tools",
    "agent"
)

# -----------------------------
# 5. Compile Agent
# -----------------------------

graph = builder.compile()

# -----------------------------
# 6. Test Agent
# -----------------------------

if __name__ == "__main__":

    result = graph.invoke(
        {
            "messages": [
                (
                    "user",
                    "What are embeddings?"
                )
            ]
        }
    )

    print("\n--- Agent Execution ---")

    for message in result["messages"]:

        print("\nMessage type:", type(message).__name__)

        if hasattr(message, "tool_calls") and message.tool_calls:
            print("🔧 Tool called:")
            print(message.tool_calls)

        if message.content:
            print("Content:")
            print(message.content)

    print("\n--- Final Answer ---")

    print(
        result["messages"][-1].content
    )
