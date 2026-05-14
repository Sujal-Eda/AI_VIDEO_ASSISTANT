from utils.audio_processing import process_input
from core.transcribing import transcribe_all
from core.summarize import summarize, generate_title
from core.extractor import extract_decisions, extract_questions, extract_action_items

source ='https://youtu.be/FLcrvMfHUJM?si=K6Z1Uv2L7iKMgIvZ'
chunks = process_input(source)
transcript =transcribe_all(chunks,True)
print(transcribe_all(chunks,True))

title = generate_title(transcript)
summary = summarize(transcript)

print("Title **********")
print(title)
print("Summary **********")
print(summary)
action_items = extract_decisions(transcript)
print("Action items **********")
print(action_items)
questions = extract_questions(transcript)
print("Questions **********")
print(questions)
decisions = extract_decisions(transcript)
print("Decisions **********")
print(decisions)