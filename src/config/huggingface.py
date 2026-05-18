from langchain_huggingface import HuggingFaceEmbeddings
from .settings import HF_EMBEDDING_MODEL


def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name=HF_EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"}, 
        encode_kwargs={"normalize_embeddings": True},
    )
