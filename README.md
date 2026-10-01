---
title: PalmScript HUD (Real-Time Sign Language Bridge)
emoji: 🤟
colorFrom: blue
colorTo: indigo
app_port: 7860
pinned: false
license: mit
---

<div align="center">

# 🤟  PalmScript HUD(Heads-Up Display)

### *Next-Gen Powered Two-Way Sign Language Conversational Bridge*

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-FF6F00.svg?logo=tensorflow&logoColor=white)](https://tensorflow.org/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-Holistic-007FFF.svg?logo=google&logoColor=white)](https://developers.google.com/mediapipe)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**An intelligent, bidirectional assistive bridge connecting the Deaf/Hard-of-Hearing community and the hearing world in real-time.**

</div>

---

## 📖 What is Echoes (PalmScript HUD)?

Over **70 million people** worldwide use sign language as their primary mode of expression, yet over **99% of hearing individuals** cannot understand or interpret sign language. This communication barrier creates daily friction in essential environments like healthcare, banking, classrooms, and everyday social interactions.

**Echoes (PalmScript HUD)** is a real-time, bidirectional assistive communication platform that eliminates this barrier completely:

1. **Sign-to-Speech (For the Signer):** The deaf user signs naturally into any standard webcam. This system extracts hand/body kinematics, recognises the gesture through deep learning, and instantly voices the word aloud via neural Text-to-Speech (TTS).
2. **Speech-to-Subtitles (For the Hearing Person):** When the hearing conversational partner speaks in reply, their voice is captured via speech recognition and transcribed live as a floating HUD subtitle on screen for the signer.
3. **Cyber-HUD Interface:** Inspired by futuristic spatial computing, the interface provides a transparent, zero-clutter heads-up display showing live skeletal tracking, prediction confidence meters, and conversational history.

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

## ⚙️ How It Works (Technical Pipeline)

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

## 📊 Model Accuracy & Benchmark

The core neural engine was rigorously evaluated across the entire 101-sign dictionary on **20,705 validation sequences**:

| Metric                             | Benchmark Result                                |
|:---------------------------------- |:----------------------------------------------- |
| **Total Sign Classes**             | **101 Signs** (Dynamic gestures & static poses) |
| **Total Test Samples**             | **20,705 Sequences** (30 frames per sequence)   |
| **Overall Model Accuracy**         | **99.99%**                                      |
| **Signs with 99% - 100% Accuracy** | **100 / 101 Signs**                             |
| **Inference Latency**              | **~15ms per frame on CPU**                      |

### Supported Vocabulary (101 Signs)

Includes everyday conversation, emergency phrases, numbers, questions, greetings, and transactions:

> *hello, thank you, sorry, please, help, love, good, ready, how are you, fine, ok, great, hi, whats-up, nothing, busy, not a lot, what's your name, nice to meet you, my name is, deaf, hearing, understand, I don't understand, slow, again, see you later, don't want, done, like, forget, i need help, more, go, happy, i, my, you, how, what, where, bathroom, want, baby, brother, boyfriend, father, mother, ask, boy, girl, work, chat, come, late, say, pay attention, tell, buy, home, car, room, walk, bread, cook, drink, ice cream, water, always, never, time, day, week, minute, hour, year, what time is it, see you next week, money, atm, bank, deposit, cash, price, free, earn, sell, tax, good bye, take care, same here, friend, introduce, coffee, tea, your name, small, medium, large.*

---

## 🌟 How This Helps (Real-World Impact)

| Domain                             | How Echoes Transforms Communication                                                                                                              |
|:---------------------------------- |:------------------------------------------------------------------------------------------------------------------------------------------------ |
| **🏥 Healthcare & Hospitals**      | Deaf patients can explain their symptoms, pain levels, and medical history directly to doctors without waiting hours for an on-site interpreter. |
| **🏦 Banking & Public Services**   | Enables independent financial transactions (deposits, ATM assistance, inquiries) at counters with complete privacy.                              |
| **🎓 Education & Classrooms**      | Deaf students can participate in classroom discussions and voice questions to professors seamlessly.                                             |
| **🤝 Everyday Social Connections** | Enables spontaneous, natural conversations between deaf individuals, friends, and family members with zero sign language experience required.    |
| **💼 Workplace Inclusivity**       | Facilitates team meetings, standups, and peer collaboration in corporate environments.                                                           |

---

## 🔮 Future Scope & Roadmap

- [ ] **Continuous Sentence Translation (Gloss-to-English LLM):** Integrating lightweight quantized LLMs (e.g. Gemma-2B / Llama-3-8B) to transform fragmented sign glosses into grammatically complete, natural English sentences.
- [ ] **AR Smart Glasses Integration:** Porting the HUD interface to wearable augmented reality headsets (Apple Vision Pro, Meta Quest, Ray-Ban Meta glasses) for real-world conversational heads-up overlays.
- [ ] **Facial Expression & Sentiment Analysis:** Utilizing MediaPipe Face Mesh to capture eyebrows, mouth shapes, and head tilts to modulate tone, emotion, and question inflection in the synthesized voice.

---

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

---

<div align="center">
Made with ❤️ for accessibility and inclusive communication.
</div>
