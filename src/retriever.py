from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.tools import create_retriever_tool
from .config import get_embeddings

def ingest_documents(url: list[str]):
    loader = WebBaseLoader(url)
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1000,
        chunk_overlap = 200
    )
    splits = text_splitter.split_documents(docs)

    embeddings = get_embeddings()

    vectorstore = FAISS.from_documents(
        documents=splits,
        embedding=embeddings
    )

    return vectorstore

def get_retriever_tools(vectorstore):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    retriever_tool = create_retriever_tool(
        retriever,
        "retrieve_document",
        "Search and retrieve relevant documents from the knowledge base. Use this when you need external information to answer the question."
    )

    return retriever_tool