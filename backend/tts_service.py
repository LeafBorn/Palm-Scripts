import asyncio
import os
import edge_tts
import hashlib

AUDIO_CACHE_DIR = os.path.join(os.path.dirname(__file__), 'audio_cache')
os.makedirs(AUDIO_CACHE_DIR, exist_ok=True)

# High quality natural neural voice
DEFAULT_VOICE = "en-US-JennyNeural"

def get_audio_filename(text: str) -> str:
    """Generate safe filename for cached audio (supports multi-word phrases)."""
    clean = "".join([c if c.isalnum() else "_" for c in text.lower().strip()])
    if len(clean) > 36:
        h = hashlib.md5(text.encode('utf-8')).hexdigest()[:8]
        return f"{clean[:28]}_{h}.mp3"
    return f"{clean}.mp3"

async def generate_speech_file(text: str, voice: str = DEFAULT_VOICE) -> str:
    """Generate audio file using edge-tts if not already cached."""
    filename = get_audio_filename(text)
    filepath = os.path.join(AUDIO_CACHE_DIR, filename)
    
    if os.path.exists(filepath) and os.path.getsize(filepath) > 0:
        return filename
    
    try:
        communicate = edge_tts.Communicate(text=text, voice=voice, rate="+10%")
        await communicate.save(filepath)
        return filename
    except Exception as e:
        print(f"⚠️ Edge-TTS generation error for '{text}': {e}")
        return ""

async def pre_cache_all_signs(signs_list):
    """Pre-generate TTS for all signs in background for 0ms latency playback."""
    print(f"🎙️ Pre-caching Edge-TTS audio for {len(signs_list)} signs...")
    cached_count = 0
    for sign in signs_list:
        filename = get_audio_filename(sign)
        filepath = os.path.join(AUDIO_CACHE_DIR, filename)
        if not (os.path.exists(filepath) and os.path.getsize(filepath) > 0):
            try:
                communicate = edge_tts.Communicate(text=sign, voice=DEFAULT_VOICE, rate="+10%")
                await communicate.save(filepath)
                cached_count += 1
            except Exception as e:
                print(f"⚠️ Error caching '{sign}': {e}")
    print(f"✅ Edge-TTS Cache ready! ({cached_count} new generated, {len(signs_list)} total ready).")

if __name__ == '__main__':
    # Test script
    async def test():
        fn = await generate_speech_file("Hello")
        print(f"Test generated: {fn}")
    asyncio.run(test())
