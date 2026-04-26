# Developing a Hybrid Defense System for IoT Physical Layer Security

<p align="center">
  <img src="https://img.shields.io/badge/Focus-IoT%20Security-blue.svg" alt="Focus">
  <img src="https://img.shields.io/badge/Technology-TinyML%20%7C%201D--CNN-green.svg" alt="Tech">
  <img src="https://img.shields.io/badge/Framework-Hybrid%20IDS%20%26%20Honeypot-orange.svg" alt="Framework">
</p>

## 🛡️ Project Overview
This project presents an advanced Hybrid Security Framework designed to protect the Physical Layer of Internet of Things (IoT) networks. By combining an Intrusion Detection System (IDS) with a proactive Honeypot mechanism, the system effectively mitigates risks associated with Signal Leakage and Smart Jamming attacks.

## 🚀 Key Technical Components

### 🧠 Intelligent Detection (TinyML & Deep Learning)
The core of the detection engine relies on state-of-the-art machine learning optimized for edge devices:
* 1D Convolutional Neural Networks (1D-CNN): Used for sophisticated feature extraction from signal data.
* TinyML Integration: The model is compressed and optimized to run directly on IoT hardware, ensuring real-time response with minimal power consumption.
* Accuracy: Achieved high precision (approx. 97%) in identifying Smart Jamming patterns.

### 🕸️ Proactive Defense (Honeypot)
* Implementation of a strategic Honeypot to lure and analyze attackers' behavior, preventing them from reaching the actual network infrastructure.
* Hybrid coordination between the Honeypot and IDS for a multi-layered defense strategy.

### 📡 Network & Signal Simulation
* Traffic Simulation: Comprehensive modeling of network traffic and physical layer signals.
* Attack Scenarios: Simulated various Smart Jamming and signal leakage environments to validate system resilience.

## 🛠️ Tech Stack & Tools
* Programming: Python (Cryptography, Signal Processing)
* Deep Learning: TensorFlow / Keras (for 1D-CNN)
* Edge AI: TinyML
* Simulation: NS-3 / NetAnim
* Environment: VS Code, Git

## 📂 Repository Structure
* 📁 src/ : Core implementation of the 1D-CNN and TinyML models.
* 📁 scripts/ : Communication protocols (AES & ECDH encryption).
* 📁 report/ : Technical Graduation Project documentation (PDF).

---
<p align="center">
  <i>Developed as part of a Graduation Project in Cybersecurity and Network Engineering.</i>
</p>