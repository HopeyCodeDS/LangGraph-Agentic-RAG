from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from .settings import (
    OPENAI_LLM_MODEL,
    OPENAI_EMBEDDING_MODEL,
    require_openai_key,
)


def get_llm(model_name: str | None = None, temperature: float = 0.0):
    return ChatOpenAI(
        model=model_name or OPENAI_LLM_MODEL,
        api_key=require_openai_key(),
        temperature=temperature,
    )


def get_embeddings():
    return OpenAIEmbeddings(
        model=OPENAI_EMBEDDING_MODEL,
        api_key=require_openai_key(),
    )
