import sys
import os
import json

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import asyncio
import cv2
import numpy as np
from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from inference import SignRecognizer
from tts_service import AUDIO_CACHE_DIR, generate_speech_file, pre_cache_all_signs

FRONTEND_DIR = os.path.join(BASE_DIR, 'frontend')

# Global Recognizer instance
recognizer: SignRecognizer = None
startup_error = ""

@asynccontextmanager
async def lifespan(app: FastAPI):
    global recognizer, startup_error
    try:
        print("🚀 Initializing SignRecognizer engine...")
        recognizer = SignRecognizer()
        print("✅ SignRecognizer ready!")
        # Trigger background audio pre-caching for 0ms TTS latency
        asyncio.create_task(pre_cache_all_signs(recognizer.signs.tolist()))
    except Exception as e:
        import traceback
        startup_error = traceback.format_exc()
        print("❌ CRITICAL ERROR IN STARTUP LIFESPAN:")
        traceback.print_exc()
    yield
    print("🛑 Shutting down SignRecognizer...")
    if recognizer:
        try:
            recognizer.close()
        except Exception:
            pass

app = FastAPI(title="PalmScript Sign Language HUD API", lifespan=lifespan)

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "recognizer_ready": recognizer is not None,
        "startup_error": startup_error
    }

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Audio cache static mount
app.mount("/audio", StaticFiles(directory=AUDIO_CACHE_DIR), name="audio")

@app.get("/api/status")
async def get_status():
    if not recognizer:
        return {"status": "loading"}
    return {
        "status": "ready",
        "signs_count": len(recognizer.signs),
        "signs": recognizer.signs.tolist(),
        "window_size": recognizer.window_size,
        "conf_threshold": recognizer.conf_threshold
    }

@app.get("/api/tts/{text:path}")
async def get_tts(text: str):
    import urllib.parse
    decoded_text = urllib.parse.unquote(text).strip()
    filename = await generate_speech_file(decoded_text)
    if filename:
        filepath = os.path.join(AUDIO_CACHE_DIR, filename)
        return FileResponse(filepath, media_type="audio/mpeg")
    return JSONResponse(status_code=500, content={"error": "TTS generation failed"})

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    global recognizer
    await websocket.accept()
    print("🔌 Client connected to WebSocket stream")
    
    if recognizer is None:
        try:
            recognizer = SignRecognizer()
        except Exception as e:
            await websocket.send_json({"error": f"Recognizer init failed: {e}"})
            
    try:
        while True:
            message = await websocket.receive()
            
            # Handle binary JPEG frame from browser webcam
            if "bytes" in message and message["bytes"]:
                raw_bytes = message["bytes"]
                np_arr = np.frombuffer(raw_bytes, np.uint8)
                frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                if frame is not None:
                    payload = recognizer.process_frame(frame)
                    await websocket.send_json(payload)
                else:
                    await websocket.send_json({"error": "Frame decode failed"})
                    
            # Handle JSON keypoints payload (if client sends pre-extracted landmarks)
            elif "text" in message and message["text"]:
                try:
                    data = json.loads(message["text"])
                    if "keypoints" in data:
                        kps = np.array(data["keypoints"], dtype=np.float32)
                        has_left = data.get("has_left", False)
                        has_right = data.get("has_right", False)
                        payload = recognizer.process_keypoints(kps, has_left, has_right)
                        await websocket.send_json(payload)
                except Exception as err:
                    await websocket.send_json({"error": str(err)})
                    
    except WebSocketDisconnect:
        print("🔌 Client disconnected from WebSocket")
    except Exception as e:
        print(f"⚠️ WebSocket error: {e}")
        try:
            await websocket.close()
        except Exception:
            pass

# Mount frontend files at root
if os.path.exists(FRONTEND_DIR):
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")

if __name__ == '__main__':
    import uvicorn
    print(f"🌐 Starting server on http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
