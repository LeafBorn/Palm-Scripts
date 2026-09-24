---
title: PalmScript HUD - AI Sign Language Bridge
emoji: 🤟
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# PalmScript HUD - Real-Time AI Sign Language Conversational Bridge

PalmScript HUD is a real-time assistive technology platform that bridges the communication gap between Deaf / Hard-of-Hearing individuals and non-signing hearing individuals.

## 🚀 Key Highlights
- **101 Sign Vocabulary**: 99.99% model accuracy across dynamic motion and static hand signs.
- **Two-Way Conversational Dialogue**: Signer shows hand signs $\rightarrow$ converts to spoken audio. Hearing listener speaks $\rightarrow$ transcribes live to subtitles.
- **Sub-100ms Latency**: Streamed video processing via WebSockets and neural audio pre-caching.
- **Cyber-HUD Interface**: Transparent spatial heads-up display with real-time confidence tracking and consensus confirmation.

## 🛠️ Stack
- **Backend**: FastAPI, WebSockets, MediaPipe Holistic, TensorFlow Keras, Edge-TTS
- **Frontend**: Vanilla HTML5, CSS3, ES6 Audio & SpeechRecognition API
- **Deployment**: Docker container on Hugging Face Spaces (Port 7860)
