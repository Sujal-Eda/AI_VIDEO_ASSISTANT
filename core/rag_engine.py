from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from core.vector_db import build_vector_store, load_vector_store, get_retriever
from core.llm import get_llm

def format_docs(docs):
    return "\n\n".join([doc.page_content for doc in docs])

def build_rag_chain(transcript:str):
    vector_store = build_vector_store(transcript)
    retriever = get_retriever(vector_store,k=4)
    llm = get_llm()

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are an expert video content analyst. Using only the transcript excerpts provided below, "
         "answer the user's question as accurately and concisely as possible. "
         "If the answer is not in the transcript, say so — do not guess. "
         "When quoting the speaker, use quotation marks.\n\n"
         "Transcript context:\n{context}"),
        ("human", "{question}"),
    ])

    rag_chain = (
        {
            "context": retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain


def load_rag_chain():
    vector_store = load_vector_store()
    retriever = get_retriever(vector_store)
    llm = get_llm()
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are an expert video content analyst. Using only the transcript excerpts provided below, "
         "answer the user's question as accurately and concisely as possible. "
         "If the answer is not in the transcript, say so — do not guess. "
         "When quoting the speaker, use quotation marks.\n\n"
         "Transcript context:\n{context}"),
        ("human", "{question}"),
    ])

    rag_chain = (
        {
            "context": retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain


def ask_question(rag_chain, question: str) -> str:
    answer = rag_chain.invoke(question)
    return answer

