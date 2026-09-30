# 🖐️ PalmScript — Real-Time AI Sign Language & Two-Way Conversational HUD

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-FF6F00.svg?logo=tensorflow&logoColor=white)](https://tensorflow.org/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-Holistic-007FFF.svg?logo=google&logoColor=white)](https://developers.google.com/mediapipe)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg?logo=docker&logoColor=white)](https://www.docker.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**An intelligent, bidirectional assistive bridge connecting the Deaf/Hard-of-Hearing community and the hearing world in real-time.**

[What is PalmScript](#-1-what-is-palmscript) • [The Idea Behind It](#-2-the-idea-behind-palmscript) • [How It Was Built](#-3-how-it-works--architecture) • [Project Goals & Vision](#-4-project-goals--future-vision) • [Quick Start](#-5-getting-started--installation) • [Deployment](#-6-cloud-deployment-render--docker)

</div>

---

## 🌟 1. What is PalmScript?

**PalmScript** is an AI-powered conversational system and futuristic Heads-Up Display (HUD) that translates **Sign Language gestures into synthesized human speech** and **spoken voice replies back into live visual subtitles** in real-time.

Unlike traditional static image classifiers, PalmScript processes **continuous temporal motion sequences** across a 101-word sign dictionary at **30+ FPS**, enabling seamless two-way dialogue without requiring human interpreters or specialized hardware gloves.

### 🔑 Key Features
* 🤟 **Real-Time Sign-to-Speech:** Tracks hand joints, wrist movement, and upper-body posture using standard webcams and speaks out recognized words/phrases using neural Text-To-Speech (Edge-TTS).
* 🗣️ **Two-Way Voice-to-Text (Talk Back):** Enables hearing conversational partners to speak naturally; their speech is transcribed live into floating subtitles for the signer.
* ⚡ **Ultra-Low Latency Sliding Window Engine:** Smooth 30-frame rolling temporal buffer enables instant detection (<25ms latency per frame) with consensus validation to eliminate false positives.
* 🌐 **Cyber-HUD Glass Interface:** Interactive Sci-Fi HUD showing real-time joint coordinates, confidence meters, token ribbons, and dual-speaker dialogue feeds.
* 🎧 **Zero-Latency Audio Pre-Caching:** High-frequency vocabulary is pre-rendered for instant playback upon gesture completion.

---

## 💡 2. The Idea Behind PalmScript

### The Problem
Over **70 million people** worldwide communicate primarily through sign language. However, more than **99% of the hearing population** cannot understand or interpret sign language. This communication gap creates daily friction:
* **Healthcare:** Deaf individuals struggle to convey emergency symptoms or understand medical instructions without an on-site interpreter.
* **Public Counters & Banking:** Routine transactions (bank deposits, ticket counters, billing) lose privacy and take excessive time.
* **Education & Employment:** Deaf students and professionals face participation barriers in meetings, classrooms, and interviews.

### The Inspiration
The vision behind **PalmScript** is to make communication **universal, spontaneous, and independent**. We asked:
> *"What if any device with a standard webcam could understand sign language as fluently as speech, acting as an omnipresent voice for the signer?"*

PalmScript eliminates the dependency on human intermediaries, giving deaf individuals total independence in daily social and professional life.

---

## ⚙️ 3. How It Works & Architecture

```mermaid
flowchart LR
    subgraph Client ["Client Browser (HUD Interface)"]
        Cam["📷 Standard Webcam (30 FPS)"] --> FramePrep["Canvas Capture & Aspect Scaling"]
        FramePrep --> WSSend["WebSocket JPEG Stream"]
        AudioOut["🔊 Neural TTS Voice"] <-- WSRecv["Prediction & Audio Stream"]
        Mic["🎙️ Person 2 (Hearing Mic)"] --> LiveSubtitles["Web Speech STT (Subtitles)"]
    end

    subgraph Server ["FastAPI & AI Backend"]
        WSSend --> WSEndpoint["WebSocket /ws Stream"]
        WSEndpoint --> MPHolistic["MediaPipe Holistic (Pose + Hands)"]
        MPHolistic --> KeypointNorm["Wrist-Relative Coordinate Normalizer (150 Features)"]
        KeypointNorm --> RollBuffer["30-Frame Continuous Sliding Buffer"]
        RollBuffer --> BiLSTM["Conv1D + Bidirectional LSTM Neural Model"]
        BiLSTM --> Consensus["Confidence (>55%) & Consensus Filter"]
        Consensus --> EdgeTTS["Microsoft Edge-TTS Neural Audio"]
        EdgeTTS --> WSResponse["JSON State Payload"]
        WSResponse --> WSRecv
    end
```

### 1. Kinematic Landmark Tracking
- Utilizes **Google MediaPipe Holistic** to extract 33 upper-body Pose landmarks and 21 landmarks per hand.
- Hand coordinates are normalized relative to each wrist origin ($x - x_{\text{wrist}}, y - y_{\text{wrist}}$), ensuring invariant detection regardless of camera distance or user position in frame.
- Total feature vector per frame: **150 spatial keypoints**.

### 2. Deep Temporal Neural Architecture
- Deep learning model combining **1D Convolutional Layers (Conv1D)** for spatial trajectory extraction with **Bidirectional Long Short-Term Memory (BiLSTM)** networks for temporal sequence classification.
- Trained on **20,700+ sequence frames** with LayerNormalization and Dropout regularization.

```
Input (30 frames × 150 features)
   │
   ├──> Conv1D (128 filters) + LayerNorm + Dropout
   ├──> Conv1D (256 filters) + LayerNorm + Dropout
   ├──> Bidirectional LSTM (256 units, return sequences) + LayerNorm
   ├──> Bidirectional LSTM (128 units, return sequences) + LayerNorm
   ├──> Bidirectional LSTM (64 units) + LayerNorm
   ├──> Dense (512) -> Dense (256) -> Dense (128)
   └──> Softmax Classifier (101 Sign Classes)
```

### 3. Continuous Sliding Window & Consensus Latching
- Uses a **30-frame rolling FIFO buffer** with temporal interpolation.
- To prevent accidental speech triggers during hand transitions, predictions undergo multi-frame probability smoothing and a single-fire speech latch.

---

## 🎯 4. Project Goals & Future Vision

* [x] **Real-time 101-word vocabulary recognition at 30+ FPS**
* [x] **Bidirectional conversation (Sign-to-Speech + Speech-to-Subtitles)**
* [x] **Zero-configuration web application (Runs on browser + webcam)**
* [ ] **Continuous Sentence Translation (Gloss-to-English LLM):** Integrating lightweight quantized LLMs (Gemma / LLaMA) to assemble individual sign tokens into grammatically complete, natural English sentences.
* [ ] **Smart AR Glasses Integration:** Porting the HUD interface into smart glasses (Ray-Ban Meta / Apple Vision Pro) for real-world conversational heads-up overlays.
* [ ] **Multi-Language Sign Support:** Expanding from ASL / ISL to British Sign Language (BSL), French Sign Language (LSF), and regional dialects.
* [ ] **On-Device Mobile Edge AI:** Compiling models to ONNX / TFLite for completely offline operation on mobile devices.

---

## 📚 5. Supported Vocabulary (101 Signs)

PalmScript comes pre-trained with 101 foundational signs across diverse domains:

> **Greetings & Social:** `hello`, `thank you`, `sorry`, `yes`, `no`, `please`, `help`, `love`, `good`, `ready`, `how are you`, `fine`, `ok`, `great`, `hi`, `whats-up`, `nothing`, `busy`, `not a lot`, `nice to meet you`, `good bye`, `take care`, `same here`, `friend`, `introduce`  
> **Identity & Questions:** `what's your name`, `my name is`, `your name`, `deaf`, `hearing`, `understand`, `I don't understand`, `slow`, `again`, `see you later`, `how`, `what`, `where`, `what time is it`  
> **Family & People:** `baby`, `brother`, `boyfriend`, `father`, `mother`, `boy`, `girl`  
> **Needs, Actions & Daily Life:** `want`, `don't want`, `done`, `like`, `forget`, `i need help`, `more`, `go`, `happy`, `i`, `my`, `you`, `work`, `chat`, `come`, `late`, `say`, `pay attention`, `tell`, `bathroom`, `home`, `car`, `room`, `walk`, `bread`, `cook`, `drink`, `ice cream`, `water`, `coffee`, `tea`, `small`, `medium`, `large`  
> **Time & Finance:** `always`, `never`, `time`, `day`, `week`, `minute`, `hour`, `year`, `see you next week`, `money`, `atm`, `bank`, `deposit`, `cash`, `price`, `free`, `earn`, `sell`, `tax`

---

## 🚀 6. Getting Started & Installation

### Prerequisites
* Python 3.10+
* Standard webcam & microphone
* Google Chrome, Microsoft Edge, or Firefox

### Installation Steps

1. **Clone the repository:**
   ```bash
   git clone https://github.com/LeafBorn/Echoes.git
   cd Echoes
   ```

2. **Create a virtual environment & install dependencies:**
   ```bash
   # On Windows
   python -m venv venv
   .\venv\Scripts\activate

   # On Linux / macOS
   python3 -m venv venv
   source venv/bin/activate

   # Install requirements
   pip install -r requirements.txt
   ```

3. **Start the application:**
   ```bash
   python backend/main.py
   # OR on Windows double-click: run_app.bat
   ```

4. Open **`http://localhost:8000`** in your browser. Allow camera and audio permissions when prompted!

---

## 🐳 7. Cloud Deployment (Render & Docker)

PalmScript includes a production-ready `Dockerfile` and `render.yaml` configuration.

### Deploying with Docker
```bash
# Build the Docker image
docker build -t palmscript-hud .

# Run the container
docker run -p 8000:10000 palmscript-hud
```

### Deploying to Render
1. Connect your repository on [Render](https://dashboard.render.com).
2. Create a new **Web Service** selecting **Docker** runtime.
3. Render will automatically build the container and deploy the app with WebSocket and HTTPS enabled.

---

## 📄 License

This project is open-source and licensed under the [MIT License](LICENSE).

<div align="center">

*Empowering voices through computer vision and accessibility.* 🖐️✨

</div>
