RhythmRay AI: Advanced Medical Diagnostic Platform

RhythmRay AI is a multi-modal medical diagnostic system built around a "System of Experts" architecture: dedicated computer vision models analyze Chest X-Rays and 12-lead ECGs, and a quantized Gemma-2B language model turns their findings into a short, readable clinical impression.

🚀 Key Features

Multi-Modal Analysis — Diagnoses both frontal Chest X-Rays and 12-lead ECG signals.
System of Experts — A DenseNet121 (modified for single-channel input) handles CXR classification, and a Keras ResNet1D model handles ECG classification.
Flexible ECG Input — Upload a raw .npy/.csv 12-lead signal for the highest accuracy, or upload a photo/scan of a printed ECG strip, which gets digitized through an OpenCV-based grid-removal and lead-extraction pipeline (flagged as approximate in the UI).
Generative Reporting — Findings are passed to google/gemma-2b-it, prompted to draft a 3–4 sentence clinical note. The model runs as-is — no LoRA fine-tuning is currently wired into the pipeline.
Clinical Ergonomics — Dark-mode Streamlit dashboard with bilingual (Arabic/English) prompts on the ECG upload flow.
Quantized Execution — Gemma-2B-it loads in 4-bit NF4 via BitsAndBytes, so it fits on a single consumer or free-tier Colab GPU.


🛠️ Architecture & Modalities

Radiology (CXR) — DenseNet121
Grayscale 224×224 input, multi-label sigmoid output over 14 conditions: Atelectasis, Cardiomegaly, Consolidation, Edema, Effusion, Emphysema, Fibrosis, Hernia, Infiltration, Mass, Nodule, Pleural Thickening, Pneumonia, Pneumothorax. Falls back to "No Finding" if nothing crosses the 0.5 threshold.

Cardiology (ECG) — ResNet1D (Keras/TensorFlow)
12-lead input (5000 samples/lead, per-lead z-score normalized), multi-label sigmoid output over the 5 PTB-XL superclasses: NORM, MI, STTC, CD, HYP. Uses per-class thresholds from thresholds_ptbxl.json when present, otherwise defaults to 0.5 for every class.

Report Generation — Gemma-2B-it
Diagnosis, confidence, and a short clinical-notes dictionary get folded into an instruction prompt, and the quantized LLM generates the final narrative report.


 **Download Model Weights:**
   * [Download CXR Model ](https://github.com/YazanAlhusseini/RhythmRay/releases/download/v1.0/ckpt_best.pt.zip
)
   * [Download ECG Model ](https://github.com/YazanAlhusseini/RhythmRay/releases/download/v1.0/best_resnet1d.keras
)
   


Unzip ckpt_best.pt.zip to get ckpt_best.pt. There's no bundled thresholds_ptbxl.json in the release — it's optional. Without it, the ECG model just falls back to a flat 0.5 threshold for all 5 classes instead of calibrated per-class thresholds.

⚙️ Running RhythmRay (Google Colab + ngrok)

app.py is currently a Colab notebook export, not a standalone local script — it installs its own dependencies with !pip install cells and expects Colab-style /content/... paths, so streamlit run app.py won't work on its own outside Colab. The flow today is:


Open app.py in Google Colab (paste its cells into a notebook, or open the file directly).

Get an ngrok authtoken and a Hugging Face token (needed to pull google/gemma-2b-it), and paste them into the NGROK_TOKEN and HF_TOKEN variables near the top of the script.

Upload ckpt_best.pt and best_resnet1d.keras (and thresholds_ptbxl.json, if you have it) into the Colab session's /content/ folder.

Run all cells. The notebook installs dependencies, downloads and quantizes Gemma-2B-it, loads both vision models, then launches Streamlit in the background and opens an ngrok tunnel.

Grab the public URL printed at the end of the run — that's the live dashboard.


requirements.txt is kept for local reference, but it's currently missing a few packages the notebook actually imports (opencv-python, tensorflow, scipy, pandas) — worth syncing if a real local setup path gets added later.

📁 Repository Structure

textRhythmRay/
├── app.py              # Colab-exported notebook: installs deps, loads models, launches the Streamlit app via ngrok
├── requirements.txt    # Reference list of Python dependencies
├── README.md           # Documentation
└── models.             # Placeholder file — model weights are downloaded from Releases, not stored here

👨‍💻 Team (Jamoum University College — CS Department)


(Team Leader)Yazan Alhusseini — ECG Expert & System Integration

Raad Aladli — LLM Fine-tuning & Web Backend

Osama Alharbi — ECG Preprocessing & Frontend Design

Thamer Alzahrani — CXR Expert & Documentation

Khaled Alsolami — CXR Preprocessing & Testing


🎓 Acknowledgments

Special thanks to Umm Al-Qura University and the College of Computer and Information Systems for their support and resources.

📜 Disclaimer

RhythmRay AI is a research prototype intended for clinical triage support. It is not intended to replace final clinical judgment by a certified healthcare professional.


© 2026 RhythmRay AI Project. All Rights Reserved.
