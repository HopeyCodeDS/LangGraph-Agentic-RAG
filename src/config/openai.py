from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from .settings import OPENAI_API_KEY

def get_llm(model_name: str = "gpt-4o-mini", temperature: float = 0.0):
    return ChatOpenAI(
        model=model_name,
        api_key=OPENAI_API_KEY,
        temperature=temperature,
    )

def get_embeddings():
    return OpenAIEmbeddings(
        api_key=OPENAI_API_KEY,
    )