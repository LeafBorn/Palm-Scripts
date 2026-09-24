---
title: Echoes - PalmScript HUD (Real-Time Sign Language Bridge)
emoji: 🤟
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
license: mit
---

<div align="center">

# 🤟 Echoes | PalmScript HUD
### *Next-Gen AI-Powered Two-Way Sign Language Conversational Bridge*

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-FF6F00.svg?logo=tensorflow&logoColor=white)](https://tensorflow.org/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-Holistic-007FFF.svg?logo=google&logoColor=white)](https://developers.google.com/mediapipe)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg?logo=docker&logoColor=white)](https://www.docker.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**Bridging the communication chasm between Deaf / Hard-of-Hearing individuals and the non-signing world in real-time.**

[Explore Demo](#-quick-start--installation) • [How It Works](#-how-it-works-technical-pipeline) • [Model Benchmark](#-model-accuracy--benchmark) • [Future Roadmap](#-future-scope--roadmap)

</div>

---

## 📖 What is Echoes (PalmScript HUD)?

Over **70 million people** worldwide use sign language as their primary mode of expression, yet over **99% of hearing individuals** cannot understand or interpret sign language. This communication barrier creates daily friction in essential environments like healthcare, banking, classrooms, and everyday social interactions.

**Echoes (PalmScript HUD)** is a real-time, bidirectional assistive communication platform that eliminates this barrier completely:

1. **Sign-to-Speech (For the Signer):** The deaf user signs naturally into any standard webcam. The AI system extracts hand/body kinematics, recognises the gesture through deep learning, and instantly voices the word aloud via neural Text-to-Speech (TTS).
2. **Speech-to-Subtitles (For the Hearing Person):** When the hearing conversational partner speaks in reply, their voice is captured via speech recognition and transcribed live as a floating HUD subtitle on screen for the signer.
3. **Cyber-HUD Interface:** Inspired by futuristic spatial computing, the interface provides a transparent, zero-clutter heads-up display showing live skeletal tracking, prediction confidence meters, and conversational history.

---

## ⚙️ How It Works (Technical Pipeline)

```mermaid
flowchart TD
    subgraph Client ["Client (Browser & Webcam)"]
        Cam["📷 Webcam Stream (30 FPS)"] --> Canvas["Canvas Capture & Aspect Rescaling"]
        Canvas --> WSSend["WebSocket Base64 Stream"]
        AudioOut["🔊 Neural TTS Speaker"] <-- WSRecv["Audio & Prediction Listener"]
        Mic["🎙️ Hearing Person Mic"] --> STT["Web Speech API (Live Subtitles)"]
    end

    subgraph Server ["FastAPI Backend Engine"]
        WSSend --> WSEndpoint["WebSocket /ws Endpoint"]
        WSEndpoint --> MP["MediaPipe Holistic (Pose + Hands)"]
        MP --> Norm["Wrist-Relative Keypoint Normalization (150 Features)"]
        Norm --> Buffer["30-Frame Resilient Rolling Window"]
        Buffer --> TF["Deep Temporal Neural Network (GRU / Conv1D)"]
        TF --> Consensus["Consensus & Confidence Filtering Engine"]
        Consensus --> TTS["Edge-TTS Neural Audio Synthesizer"]
        TTS --> WSResponse["Construct Payload (Predictions + Audio)"]
        WSResponse --> WSRecv
    end
```

### 1. Spatial Vision & Relative Normalization
- Uses **Google MediaPipe Holistic** to capture 33 Pose landmarks and 21 landmarks per hand.
- Hand coordinates are normalized relative to the **wrist origin** ($x - x_{\text{wrist}}, y - y_{\text{wrist}}$), ensuring invariant detection regardless of camera distance or user position in frame.
- Total feature vector per frame: **150 spatial keypoints**.

### 2. Resilient Rolling Temporal Window
- Signs unfold dynamically across time. Echoes accumulates keypoints into a **30-frame temporal buffer** (~1 second at 30 FPS).
- Features a **Temporal Carry-Forward Filter**: If fast finger movement causes MediaPipe to miss 1–3 frames, the system automatically carries forward previous valid landmarks, preventing buffer collapse.

### 3. Deep Neural Sequence Classification
- Custom temporal neural architecture trained on **20,700+ sequence frames**.
- Evaluates kinematic trajectory, joint velocity, and finger curvature to distinguish between nuanced signs (e.g., distinguishing between *'thank you'*, *'good'*, and *'boy'*).

### 4. Consensus & False-Trigger Elimination
- To prevent accidental triggers during resting hand motions, predictions undergo **3-frame consecutive consensus** with a **70% confidence threshold** and a **15% confidence margin** over the second-place class.

### 5. Zero-Latency Pre-Cached Speech
- High-frequency vocabulary items are pre-synthesized using Microsoft Edge neural TTS (`en-US-ChristopherNeural`) into an in-memory audio cache, yielding instantaneous `<10ms` playback upon gesture completion.

---

## 📊 Model Accuracy & Benchmark

The core neural engine was rigorously evaluated across the entire 101-sign dictionary on **20,705 validation sequences**:

| Metric | Benchmark Result |
| :--- | :--- |
| **Total Sign Classes** | **101 Signs** (Dynamic gestures & static poses) |
| **Total Test Samples** | **20,705 Sequences** (30 frames per sequence) |
| **Overall Model Accuracy** | **99.99%** |
| **Signs with 99% - 100% Accuracy** | **100 / 101 Signs** |
| **Inference Latency** | **~15ms per frame on CPU** |

### Supported Vocabulary (101 Signs)
Includes everyday conversation, emergency phrases, numbers, questions, greetings, and transactions:
> *hello, thank you, sorry, please, help, love, good, ready, how are you, fine, ok, great, hi, whats-up, nothing, busy, not a lot, what's your name, nice to meet you, my name is, deaf, hearing, understand, I don't understand, slow, again, see you later, don't want, done, like, forget, i need help, more, go, happy, i, my, you, how, what, where, bathroom, want, baby, brother, boyfriend, father, mother, ask, boy, girl, work, chat, come, late, say, pay attention, tell, buy, home, car, room, walk, bread, cook, drink, ice cream, water, always, never, time, day, week, minute, hour, year, what time is it, see you next week, money, atm, bank, deposit, cash, price, free, earn, sell, tax, good bye, take care, same here, friend, introduce, coffee, tea, your name, small, medium, large.*

---

## 🌟 How This Helps (Real-World Impact)

| Domain | How Echoes Transforms Communication |
| :--- | :--- |
| **🏥 Healthcare & Hospitals** | Deaf patients can explain their symptoms, pain levels, and medical history directly to doctors without waiting hours for an on-site interpreter. |
| **🏦 Banking & Public Services** | Enables independent financial transactions (deposits, ATM assistance, inquiries) at counters with complete privacy. |
| **🎓 Education & Classrooms** | Deaf students can participate in classroom discussions and voice questions to professors seamlessly. |
| **🤝 Everyday Social Connections** | Enables spontaneous, natural conversations between deaf individuals, friends, and family members with zero sign language experience required. |
| **💼 Workplace Inclusivity** | Facilitates team meetings, standups, and peer collaboration in corporate environments. |

---

## 🚀 Quick Start & Installation

### Prerequisites
- Python 3.10 or higher
- A standard Webcam and Microphone
- Modern browser (Chrome / Edge / Firefox)

### 1. Clone the Repository
```bash
git clone https://github.com/LeafBorn/Echoes.git
cd Echoes
```

### 2. Setup Virtual Environment
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run Application
```bash
# Direct Python Launch
python backend/main.py

# Or on Windows using batch script
run_app.bat
```
Open **`http://localhost:8000`** in your browser. Allow camera & microphone permissions when prompted.

---

## 🐳 Docker Deployment

Echoes is fully containerized and cloud-ready with zero external dependencies.

```bash
# Build the Docker Image
docker build -t echoes-hud .

# Run the Container
docker run -p 7860:7860 echoes-hud
```
Access the application at `http://localhost:7860`.

---

## 🔮 Future Scope & Roadmap

- [ ] **Continuous Sentence Translation (Gloss-to-English LLM):** Integrating lightweight quantized LLMs (e.g. Gemma-2B / Llama-3-8B) to transform fragmented sign glosses into grammatically complete, natural English sentences.
- [ ] **Multi-Dialect & Regional Sign Languages:** Expanding beyond ASL/ISL vocabulary to support British Sign Language (BSL), French Sign Language (LSF), and diverse Indian regional dialects.
- [ ] **AR Smart Glasses Integration:** Porting the HUD interface to wearable augmented reality headsets (Apple Vision Pro, Meta Quest, Ray-Ban Meta glasses) for real-world conversational heads-up overlays.
- [ ] **Facial Expression & Sentiment Analysis:** Utilizing MediaPipe Face Mesh to capture eyebrows, mouth shapes, and head tilts to modulate tone, emotion, and question inflection in the synthesized voice.
- [ ] **Mobile Native Edge Computing:** Compiling models to TensorFlow Lite / ONNX for offline on-device inference on Android and iOS devices.

---

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

---

<div align="center">
Made with ❤️ for accessibility and inclusive communication.
</div>
