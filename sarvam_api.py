import os
import io
import wave
import base64
import textwrap
import json
from sarvamai import SarvamAI
from dotenv import load_dotenv

load_dotenv()

client = SarvamAI(
    api_subscription_key=os.getenv("SARVAM_API_KEY")
)

def combine_wav_bytes(wav_list):
    """Safely stitches WAV bytes. Falls back to first chunk if stitching fails."""
    if not wav_list:
        return None
    if len(wav_list) == 1:
        return wav_list[0]

    output = io.BytesIO()
    try:
        with wave.open(io.BytesIO(wav_list[0]), 'rb') as first_wav:
            params = first_wav.getparams()
            with wave.open(output, 'wb') as out_wav:
                out_wav.setparams(params)
                out_wav.writeframes(first_wav.readframes(first_wav.getnframes()))
                for wav_bytes in wav_list[1:]:
                    try:
                        with wave.open(io.BytesIO(wav_bytes), 'rb') as next_wav:
                            if next_wav.getparams()[:3] == params[:3]:
                                out_wav.writeframes(next_wav.readframes(next_wav.getnframes()))
                    except:
                        continue
        return output.getvalue()
    except:
        return wav_list[0]

def extract_base64(response):
    """
    Extracts base64 audio from Sarvam response.
    Target field is 'audios' (list), usually the first element.
    """
    # 1. Check if it's a Pydantic Object (SDK standard)
    if hasattr(response, "audios") and response.audios:
        return response.audios[0]
    
    # 2. Check if it's a Dictionary (Raw JSON)
    if isinstance(response, dict):
        if "audios" in response and len(response["audios"]) > 0:
            return response["audios"][0]
    
    # 3. Fallback: Check for singular 'audio' just in case
    if hasattr(response, "audio") and response.audio:
        return response.audio
        
    return None

def generate_sarvam_tts(text: str, language: str):
    clean_text = text.replace("*", "").replace("#", "").strip()
    chunks = textwrap.wrap(clean_text, width=450, break_long_words=False)
    
    print(f"Processing Text: {len(clean_text)} chars -> {len(chunks)} chunks")
    
    audio_segments = []

    for i, chunk in enumerate(chunks):
        if not chunk.strip(): continue
            
        try:
            print(f"Fetching chunk {i+1}/{len(chunks)}...")
            response = client.text_to_speech.convert(
                text=chunk,
                target_language_code="en-IN",
                model="bulbul:v2",
                speaker="anushka"
            )
            
            # Use the fixed extractor that looks for 'audios'
            b64_string = extract_base64(response)
            
            if b64_string:
                chunk_bytes = base64.b64decode(b64_string)
                audio_segments.append(chunk_bytes)
            else:
                print(f"⚠️ Chunk {i+1} extraction failed. Response keys: {dir(response)}")

        except Exception as e:
            print(f"❌ Failed chunk {i+1}: {e}")

    if not audio_segments:
        print("❌ No audio segments were generated.")
        return None

    return combine_wav_bytes(audio_segments)