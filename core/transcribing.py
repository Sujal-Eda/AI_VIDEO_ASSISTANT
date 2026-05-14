import whisper
import os

WHISPER_MODEL = os.getenv('WHISPER_MODEL',"small")

_model = None

def load_model():
    global _model
    if _model is None:
        print("loading model...")
        _model = whisper.load_model(WHISPER_MODEL)
        print(f"model loaded ({WHISPER_MODEL})")
    return _model

def transcribe_chunk(chunk_path :str , translate : bool = False) -> str:
    model = load_model()
    task = 'translate' if translate else 'transcribe'
    result = model.transcribe(chunk_path, task=task)
    return result['text']

def transcribe_all(chunks: list, translate: bool = False) -> str:
    parts = []
    for i, chunk in enumerate(chunks):
        print(f"Transcribing chunk {i+1}/{len(chunks)}...")
        parts.append(transcribe_chunk(chunk, translate=translate))
    print("All chunks transcribed.")
    return "".join(parts)
    
    



