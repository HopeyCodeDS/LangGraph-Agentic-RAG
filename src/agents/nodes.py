from typing import Annotated
from langchain_core.messages import HumanMessage, SystemMessage
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

    def rewrite(state):
        messages = state["messages"]
        question = messages[0].content

        rewrite_prompt = f"""You are a query rewriter. Your task is to transform the user's question into a more effective search query.
        
                                Original question: {question}

                                Rewrite this question to be more search-friendly by:
                                1. Using more precise terminology
                                2. Adding relevant keywords
                                3. Making it clearer and more specific

                                Return ONLY the rewritten question, nothing else."""
        
        response = llm.invoke([SystemMessage(content=rewrite_prompt)])

        return {"messages": [HumanMessage(content=response.content)]}
    
    return rewrite

def create_generate_node():
    llm = get_llm()

    def generate(state):
        messages = state["messages"]
        question = messages[0].content

        context = ""
        for msg in messages:
            if hasattr(msg, 'content') and isinstance(msg.content, str):
                if "Retrieved documents" in msg.content or "Document" in msg.content:
                    context = msg.content
                    break
        
        generation_prompt = f"""You are an AI assistant. Answer the question based on the provided context.

                                Context: {context}

                                Question: {question}

                                Provide a clear, accurate answer based solely on the context provided."""

        response = llm.invoke([SystemMessage(content=generation_prompt)])
        
        return {"messages": [response]}
    
    return generate