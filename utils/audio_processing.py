import yt_dlp

from pydub import AudioSegment
import ffmpeg
import os

Download_Dir = 'downloads'
os.makedirs(Download_Dir, exist_ok=True)

def download_youtube_audio(url :str) -> str:
    output = os.path.join(Download_Dir, "%(title)s.%(ext)s")
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": output,
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "mp3",
            "preferredquality": "192",
        }],
        "quiet": True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        filename = ydl.prepare_filename(info).replace(".webm", ".mp3").replace(".m4a", ".mp3")

    return filename



def convert_to_wav(filename :str) -> str:
    output_path = os.path.splitext(filename)[0]+"_converted.wav"
    audio = AudioSegment.from_file(filename)
    audio = audio.set_channels(1).set_frame_rate(16000) #conveting to 16khz
    audio.export(output_path, format="wav")
    return output_path


def chunk_audio(wav_path :str, chunk_minutes :int = 10) -> list:
    audio = AudioSegment.from_wav(wav_path)
    chunks_ms = chunk_minutes * 60 * 1000
    chunks = []
    for i, start in enumerate(range(0, len(audio), chunks_ms)):
        chunk = audio[start : start + chunks_ms]
        chunk_path = f'{wav_path}_chunk_{i}.wav'
        chunk.export(chunk_path, format="wav")
        chunks.append(chunk_path)
    return chunks

def process_input(source: str) -> list:
    if source.startswith("http://") or source.startswith("https://"):
        print("Downloading audio from YouTube...")
        wav_path = download_youtube_audio(source)
        print("Converting audio to WAV...")
        wav_path = convert_to_wav(wav_path)
    else:
        print("Detected local audio file.")
        wav_path = convert_to_wav(source)

    print("Chunking audio...")
    chunks = chunk_audio(wav_path)
    print(f"Audio chunked into {len(chunks)} files.")
    return chunks



# data = download_youtube_audio('https://youtu.be/EdZWPB1fIJc?si=xDrlTZqkwvqKpStC')
# data_final = convert_to_wav(data)
# print(chunk_audio(data_final,5))
# # print('*' * 20 +'audio processing ran successfully ' + '*' * 20)
# data = process_input('https://youtu.be/EdZWPB1fIJc?si=xDrlTZqkwvqKpStC')
# print('*' * 20 +'audio processing ran successfully ' + '*' * 20)