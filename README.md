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

⚙️ Installation & Setup
Clone the repository:

Bash
git clone [https://github.com/YazanAlhusseini/RhythmRay.git](https://github.com/YazanAlhusseini/RhythmRay.git)
cd RhythmRay-AI
Install dependencies:

Bash
pip install -r requirements.txt
3. **Download Model Weights:**
   * [Download CXR Model (270MB)](sha256:6b54a7485005ad34e21e0b9e7d1c334d570abd666097013b960a553fc0ab2a91)
   * [Download ECG Model (16MB)](رابط_ملف_تخطيط_القلب_الذي_نسخته)
   * Create a folder named `models` in the root directory and place these files inside it.

Place both files inside the /models/ directory.

Configuration:

Open app.py and ensure the MODEL_DIR path points to your models/ folder.

Provide your HuggingFace Token in the sidebar when prompted to enable the MedGemma LLM.

Run the Application:

Bash
streamlit run app.py
🧠 Model Optimization Details
Vision: ResNet50 (Transfer Learning) & EfficientNet-B0.

LLM: MedGemma-2B (Google Gemma-2B-it base).

Fine-tuning: LoRA (Low-Rank Adaptation).

Quantization: BitsAndBytes 4-bit NormalFloat (NF4).

👨‍💻 Team (Jamoum University College - CS Department)
Yazan Alhusseini - ECG Expert & System Integration

Raad Aladli - LLM Fine-tuning & Web Backend

Osama Alharbi - ECG Preprocessing & Frontend Design

Thamer Alzahrani - CXR Expert & Documentation

Khaled Alsolami - CXR Preprocessing & Testing

📜 Disclaimer
RhythmRay AI is a research prototype designed for clinical triage support. It is not intended to replace final clinical judgment by a certified healthcare professional.
