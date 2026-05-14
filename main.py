from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
from utils.audio_processing import process_input
from core.transcribing import transcribe_all
from core.summarize import summarize, generate_title
from core.extractor import extract_key_topics, extract_key_quotes, extract_takeaways
from core.rag_engine import build_rag_chain, ask_question, load_rag_chain

load_dotenv()

def run_pipeline(source: str):
    print("Starting Video Assistant...")

    chunks = process_input(source)
    transcript = transcribe_all(chunks)
    print(f"Transcript (first 300 chars): {transcript[:300]}\n")

    tasks = {
        "title":      lambda: generate_title(transcript),
        "summary":    lambda: summarize(transcript),
        "topics":     lambda: extract_key_topics(transcript),
        "quotes":     lambda: extract_key_quotes(transcript),
        "takeaways":  lambda: extract_takeaways(transcript),
    }

    results = {}
    print("Analyzing content (running in parallel)...")
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(fn): key for key, fn in tasks.items()}
        for future in as_completed(futures):
            key = futures[future]
            results[key] = future.result()
            print(f"  [{key}] done")

    rag_chain = build_rag_chain(transcript)

    return {
        "title":      results["title"],
        "transcript": transcript,
        "summary":    results["summary"],
        "topics":     results["topics"],
        "quotes":     results["quotes"],
        "takeaways":  results["takeaways"],
        "rag_chain":  rag_chain,
    }


if __name__ == "__main__":
    source = "https://youtu.be/7kEmjETlJVE?si=ZCtUftPZDttPl9OQ"
    result = run_pipeline(source)

    sep = "=" * 60
    print(f"\n{sep}")
    print(f"TITLE:      {result['title']}")
    print(f"\n{sep}\nSUMMARY\n{sep}")
    print(result["summary"])
    print(f"\n{sep}\nKEY TOPICS\n{sep}")
    print(result["topics"])
    print(f"\n{sep}\nKEY QUOTES\n{sep}")
    print(result["quotes"])
    print(f"\n{sep}\nTAKEAWAYS\n{sep}")
    print(result["takeaways"])
    print(sep)

    print("\nChat with your video — type 'exit' to quit")
    rag_chain = result["rag_chain"]
    while True:
        question = input("YOU: ").strip()
        if question.lower() == "exit":
            print("Goodbye!")
            break
        if not question:
            continue
        answer = ask_question(rag_chain, question)
        print(f"AI: {answer}\n")
