from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnableLambda
from core.llm import get_llm

def split_transcript(transcript: str) -> list:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size = 3000,
        chunk_overlap = 200,

    )
    return splitter.split_text(transcript)#object ko bhitra function xa

def summarize(transcript: str) -> str:
    llm = get_llm()
    map_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are an expert content analyst. Summarize the following excerpt from a video transcript "
         "precisely and concisely, preserving the key points and any important details:"),
        ("human", "{text}"),
    ])

    map_chain = map_prompt | llm | StrOutputParser()

    chunks = split_transcript(transcript)
    chunk_summaries = [map_chain.invoke({"text": chunk}) for chunk in chunks]
    combined = "\n\n".join(chunk_summaries)

    combined_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are an expert content analyst. Combine the following partial summaries of a video "
         "into a single, well-structured summary. Use bullet points for key points followed by "
         "2-3 sentences of overall context. Do not repeat yourself."),
        ("human", "{text}"),
    ])

    combined_chain = (
        RunnableLambda(lambda x: {"text": x}) | combined_prompt | llm | StrOutputParser()
    )

    return combined_chain.invoke(combined)
                                 


def generate_title(transcript: str) -> str:
    llm = get_llm()
    
    title_chain = (
        RunnableLambda(lambda x: {"text": x})
        | ChatPromptTemplate.from_messages([
            ("system",
             "You are an expert content analyst. Generate a clear, descriptive title for this video "
             "based on its transcript. The title should reflect the actual content — not be generic. "
             "Return only the title, nothing else."),
            ("human", "{text}"),
        ])
        | llm
        | StrOutputParser()
    )

    snippet = transcript[:3000]
    return title_chain.invoke(snippet).strip()
                                
                                 