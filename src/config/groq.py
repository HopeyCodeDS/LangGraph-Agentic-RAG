from langchain_groq import ChatGroq
from .settings import GROQ_MODEL, require_groq_key


def get_llm(model_name: str | None = None, temperature: float = 0.0):
    return ChatGroq(
        model=model_name or GROQ_MODEL,
        api_key=require_groq_key(),
        temperature=temperature,
    )
