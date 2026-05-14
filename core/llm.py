import os
from langchain_mistralai import ChatMistralAI

def get_llm() -> ChatMistralAI:
    return ChatMistralAI(
        model="mistral-small-latest",
        mistral_api_key=os.getenv("MISTRAL_API_KEY"),
        temperature=0.3,
    )
