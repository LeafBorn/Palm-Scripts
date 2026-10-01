import sys
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import cv2
import base64
import asyncio
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

@asynccontextmanager
async def lifespan(app: FastAPI):
    global recognizer
    print("🚀 Initializing SignRecognizer engine...")
    recognizer = SignRecognizer()
    # Trigger background audio pre-caching for 0ms TTS latency
    asyncio.create_task(pre_cache_all_signs(recognizer.signs.tolist()))
    
    # Auto-open browser when model is ready
    async def open_browser():
        await asyncio.sleep(1.0)
        import webbrowser
        current_port = os.environ.get("PORT", "8000")
        webbrowser.open(f"http://localhost:{current_port}")
        print(f"🌐 Browser opened to http://localhost:{current_port}")
        
    asyncio.create_task(open_browser())
    yield
    print("🛑 Shutting down SignRecognizer...")
    if recognizer:
        recognizer.close()

app = FastAPI(title="PalmScript Sign Language HUD API", lifespan=lifespan)

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
    await websocket.accept()
    print("🔌 Client connected to Sign Recognition WebSocket stream")
    try:
        while True:
            # Receive frame as base64 or bytes
            data = await websocket.receive_text()
            
            # Format: 'data:image/jpeg;base64,...' or raw base64 string
            if ',' in data:
                _, encoded = data.split(',', 1)
            else:
                encoded = data
                
            img_bytes = base64.b64decode(encoded)
            np_arr = np.frombuffer(img_bytes, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            
            if frame is None:
                await websocket.send_json({"error": "Invalid frame"})
                continue
                
            # Process with SignRecognizer
            payload = recognizer.process_frame(frame)
            await websocket.send_json(payload)
            
    except WebSocketDisconnect:
        print("🔌 Client disconnected from WebSocket")
    except Exception as e:
        print(f"⚠️ WebSocket error: {e}")
        try:
            await websocket.close()
        except:
            pass

# Mount frontend files at root
if os.path.exists(FRONTEND_DIR):
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")

if __name__ == '__main__':
    import uvicorn
    import argparse
    
    parser = argparse.ArgumentParser(description="PalmScript Sign Language Recognition Server")
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", 8000)), help="Port to run server on")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host address (default: 0.0.0.0)")
    args, _ = parser.parse_known_args()
    
    os.environ["PORT"] = str(args.port)
    print(f"🌐 Starting server on http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)

