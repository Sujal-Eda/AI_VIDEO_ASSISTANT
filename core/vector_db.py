import os
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import torch

CHROMA_DIR = "vector_db"
COLLECTION_NAME = "video_transcript"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

_embeddings = None

def get_embeddings() -> HuggingFaceEmbeddings:
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
        )
    return _embeddings

def build_vector_store(transcript:str) -> Chroma:
    print("Building vector store...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size = 500,
        chunk_overlap = 100,
    )
    chunks = splitter.split_text(transcript)
    ## convert these chunks into vector

    docs = [
        Document(page_content = chunk, metadata = {'chunk_index': i}) for i, chunk in enumerate(chunks)
    ]
    #chunks converted into document
    # Now embedding

    embeddings = get_embeddings()
    vector_store = Chroma.from_documents(
        documents = docs,
        embedding = embeddings,
        persist_directory = CHROMA_DIR,
        collection_name = COLLECTION_NAME
    )

    return vector_store


def load_vector_store() -> Chroma:
    print("Loading vector store...")
    embeddings = get_embeddings()
    vector_store = Chroma(
        persist_directory = CHROMA_DIR,
        embedding_function = embeddings,
        collection_name = COLLECTION_NAME
    )
    return vector_store

def get_retriever(vector_store: Chroma, k: int = 4) :
    return vector_store.as_retriever(
        search_type = 'similarity',
        search_kwargs = {
            'k': k
        }
    )



