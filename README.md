# RhythmRay AI: Advanced Medical Diagnostic Platform

RhythmRay AI is a multi-modal medical diagnostic system that leverages a **Modular "System of Experts" Architecture**. It combines task-specific Computer Vision models with a fine-tuned Large Language Model (MedGemma) to analyze medical imagery (Chest X-Rays and ECGs) and synthesize professional clinical reports.

## 🚀 Key Features
- **Multi-Modal Analysis:** Supports both 12-lead ECG signals and Frontal Chest X-Rays.
- **System of Experts:** Uses specialized CNNs (ResNet50 & EfficientNet-B0) for high-precision visual extraction.
- **Generative Reporting:** Integrated MedGemma-2B (Fine-tuned via LoRA) for human-readable clinical impressions.
- **Clinical Ergonomics:** Dark-mode Streamlit dashboard designed for radiology reading rooms.
- **Quantized Execution:** Optimized for 4-bit (NF4) quantization to run efficiently on consumer-grade and cloud GPUs.

## 🛠️ Architecture & Modalities
The system classifies 13 pulmonary diseases and 5 core cardiac arrhythmias:
- **Radiology (CXR):** Atelectasis, Cardiomegaly, Effusion, Infiltration, Mass, Nodule, Pneumothorax, etc.
- **Cardiology (ECG):** Normal Sinus Rhythm (NORM), CLBBB, CRBBB, PACE, and PVC.

## 📁 Repository Structure
```text
RhythmRay-AI/
├── app.py              # Main Streamlit Application Core
├── requirements.txt    # Required Python Libraries
├── README.md           # Documentation
└── models/             # Local directory for model weights (.pth files)

## 🎓 Acknowledgments

Special thanks to **Umm Al-Qura University** and the **College of Computer and Information Systems** for their support and resources.

---
*© 2026 RhythmRay AI Project. All Rights Reserved.*
