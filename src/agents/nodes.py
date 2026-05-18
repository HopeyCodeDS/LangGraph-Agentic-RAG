from typing import Annotated
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langgraph.prebuilt import ToolNode
from ..config import get_llm

def create_agent_node(tools):
    llm = get_llm()
    llm_with_tools = llm.bind_tools(tools)

    def agent(state):
        messages = state["messages"]
        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}
    
    
    return agent

def create_rewrite_node():
    llm = get_llm()

    system_prompt = (
        "You are a query rewriter. Transform the user's question into a more "
        "effective search query by:\n"
        "1. Using more precise terminology\n"
        "2. Adding relevant keywords\n"
        "3. Making it clearer and more specific\n\n"
        "Return ONLY the rewritten question, nothing else."
    )

    def rewrite(state):
        messages = state["messages"]
        question = messages[0].content

        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Rewrite this question for search:\n\n{question}"),
        ])

        return {
            "messages": [HumanMessage(content=response.content)],
            "rewrites": state.get("rewrites", 0) + 1,
        }

    return rewrite

def create_generate_node():
    llm = get_llm()

    system_prompt = (
        "You are a helpful AI assistant. Answer the user's question using ONLY "
        "the provided context. If the context does not contain the answer, say so."
    )

    def generate(state):
        messages = state["messages"]
        question = messages[0].content

        context = ""
        for msg in reversed(messages):
            if isinstance(msg, ToolMessage):
                context = msg.content
                break

        user_prompt = (
            f"Context:\n{context}\n\n"
            f"Question: {question}\n\n"
            "Provide a clear, accurate answer based solely on the context above."
        )

        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ])

        return {"messages": [response]}

    return generate