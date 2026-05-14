AI Video Assistant with RAG

Paste a YouTube URL and the app will transcribe the audio, extract key topics, quotes, and takeaways, then let you ask questions about the content through a chat interface.

Transcription runs locally using OpenAI Whisper. Text analysis and chat use the Mistral API. The RAG pipeline stores transcript embeddings in a local ChromaDB database so you can ask follow-up questions without re-processing the video.


Requirements

- Python 3.10 or higher
- FFmpeg installed on your system
- A Mistral API key (free tier available at https://console.mistral.ai)

FFmpeg installation:
- Ubuntu / Debian: sudo apt install ffmpeg
- Windows: download from https://ffmpeg.org/download.html and add it to your PATH


Setup

1. Clone the repository

    git clone https://github.com/Sujal-Eda/AI_VIDEO_ASSISTANT.git
    cd ai-video-assistant

2. Create a virtual environment and install dependencies

    python -m venv venv
    source venv/bin/activate       
    pip install -r requirements.txt

3. Create a .env file in the project root with your Mistral API key

    MISTRAL_API_KEY=your_key_here

4. Run the app

    streamlit run app.py

Then open http://localhost:8501 in your browser.


How to use

- Paste a YouTube URL into the input field and click Analyze.
- The app downloads the audio, transcribes it, and runs analysis. This can take several minutes depending on video length and your hardware.
- Once done, browse the Summary, Key Topics, Key Quotes, and Takeaways tabs.
- Use the chat box at the bottom to ask specific questions about the video.
- Click Clear to reset and analyze a different video.


How it works

- Audio is downloaded with yt-dlp and converted to 16kHz mono WAV using pydub.
- Long audio is split into 10-minute chunks and transcribed with Whisper (small model by default).
- The transcript is embedded using the all-MiniLM-L6-v2 model from HuggingFace and stored in ChromaDB.
- Analysis tasks (summary, topics, quotes, takeaways) run in parallel using Mistral AI via LangChain.
- The chat interface retrieves the most relevant transcript chunks for each question before generating an answer.


Notes

- The first run will download the Whisper model and the embedding model automatically.
- GPU acceleration is used automatically if CUDA is available.
- To use a larger Whisper model for better accuracy, set WHISPER_MODEL=medium or WHISPER_MODEL=large in your .env file.
- Local audio files (MP3, WAV, etc.) are also supported. Pass the file path instead of a URL when using main.py directly.
