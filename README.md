# 🧠 Pupil & BPM-Based Psychological State Analyzer

## 📘 Overview

This project introduces a novel method to assess psychological and physiological conditions by analyzing two key biometric indicators:

- **Eye Pupil Diameter** (captured using MediaPipe and OpenCV)
- **Heart Rate (BPM)** (captured via a smartwatch using Bluetooth Low Energy)

The system combines these two streams of data in real time to detect signs of:

- Mental stress
- Fatigue
- Deception
- Cognitive overload
- Potential mental health conditions

All of this can be integrated into modern **smart glasses** or **wearables**.

---

## 🔍 Core Idea

> Leverage pupil dilation and heart rate patterns to detect risk factors related to stress, attention, and emotional state.

This system can:

- Predict behavioral anomalies
- Flag psychological red flags
- Act as a passive mental health monitor

---

## 🧪 Tech Stack

| Component             | Stack Used                                 |
| --------------------- | ------------------------------------------ |
| Eye/Pupil Detection   | OpenCV + MediaPipe                         |
| Heart Rate Monitoring | BLE (Bluetooth Low Energy) + Bleak Library |
| Language              | Python                                     |
| Real-Time Interface   | OpenCV GUI or Smart Glasses Integration    |
| Deployment Target     | Wearables, Desktop, Smart Glasses          |

---

## ⚙️ How It Works

1. **Capture Video**: Webcam/smart glasses provide face feed.
2. **Extract Pupil Data**: Use MediaPipe to get iris landmarks and compute diameter.
3. **Fetch BPM**: Use smartwatch BLE signal to get real-time heart rate.
4. **Combine Metrics**: Cross-analyze pupil size and BPM to detect patterns.
5. **Trigger Alerts**: Based on thresholds or ML models, notify users.

---

## ✅ Use Cases (One-liners)

- **Workplace Stress Management**
- **Mental Health Therapy & Monitoring**
- **Lie Detection & Security Screening**
- **Driver Fatigue and Attention Monitoring**
- **Remote Patient Monitoring**

---

## 🎯 Objectives

- Real-time pupil diameter tracking using computer vision
- Real-time heart rate fetching from BLE smartwatch
- Synchronize both data streams
- Apply logic or ML to analyze state/stress levels
- Give real-time user feedback (text, color, or audio)

---

## 🚀 Future Scope

- Integrate LSTM/CNN-LSTM models for temporal predictions
- Export data to cloud DB for long-term analysis
- Sync alerts to mobile app or smart glasses HUD
- Use with real-life conditions like test anxiety, fatigue in factory workers, or pilot attention tracking

---

## 📦 File Structure (Planned)

```
project/
├── preprocessing/
│   └── heart_beat/
│       └── api_fetch.py       # BLE BPM reader class
│
├── eye_tracking/
│   └── pupil_detector.py      # MediaPipe-based pupil detector class
│
├── main.py                    # Syncs both data streams
└── README.md
```

---

## 💡 Credits

Developed with a vision to bring AI-powered biometric monitoring into everyday wearables to help users stay mentally and physically fit.

> Let’s bridge the gap between tech and human wellness!

