from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from core.llm import get_llm


def build_chain(system_prompt: str):
    llm = get_llm()
    return (
        RunnablePassthrough()
        | RunnableLambda(lambda x: {"text": x})
        | ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{text}"),
        ])
        | llm
        | StrOutputParser()
    )


def extract_key_topics(transcript: str) -> str:
    chain = build_chain(
        "You are an expert content analyst. Given the transcript of a video, "
        "identify the main topics and themes discussed. For each topic provide:\n"
        "- Topic name\n"
        "- Brief description (1-2 sentences)\n"
        "- Approximate portion of the video it covers (beginning / middle / end)\n"
        "Format as a numbered list. Be specific — avoid generic labels like 'Introduction'."
    )
    return chain.invoke(transcript)


def extract_key_quotes(transcript: str) -> str:
    chain = build_chain(
        "You are an expert content analyst. Given the transcript of a video, "
        "extract the most notable, insightful, or memorable quotes or statements. "
        "For each quote provide:\n"
        "- The exact quote (in quotation marks)\n"
        "- Why it is significant or what point it illustrates\n"
        "Format as a numbered list. If no standout quotes exist, say 'No notable quotes found'."
    )
    return chain.invoke(transcript)


def extract_takeaways(transcript: str) -> str:
    chain = build_chain(
        "You are an expert content analyst. Given the transcript of a video, "
        "extract the key takeaways or action items for the viewer. For each takeaway provide:\n"
        "- The takeaway or action\n"
        "- Why it matters (one sentence)\n"
        "Format as a numbered list. If none found, say 'No clear takeaways found'."
    )
    return chain.invoke(transcript)
