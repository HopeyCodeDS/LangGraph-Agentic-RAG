from langchain_google_genai import ChatGoogleGenerativeAI
from .settings import GEMINI_MODEL, require_google_key


def get_llm(model_name: str | None = None, temperature: float = 0.0):
    return ChatGoogleGenerativeAI(
        model=model_name or GEMINI_MODEL,
        google_api_key=require_google_key(),
        temperature=temperature,
    )
