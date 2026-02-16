# 🫀 RhythmRay AI: Advanced Diagnostic System

**RhythmRay** is a state-of-the-art medical diagnostic system developed by senior Computer Science students at **Umm Al-Qura University**. The system leverages Large Language Models (LLMs) and Computer Vision to analyze medical data and provide accurate, real-time diagnoses for cardiac and pulmonary conditions.

---

## 🚀 Project Overview

The core of RhythmRay is built upon **MedGemma-2B**, a specialized version of Google's Gemma model, fine-tuned using **LoRA (Low-Rank Adaptation)** techniques. This allows the system to act as an intelligent medical assistant capable of:
1.  **Chest X-Ray Analysis:** Detecting pneumonia and other pulmonary abnormalities.
2.  **ECG Signal Processing:** Analyzing electrocardiogram waveforms to identify arrhythmias (e.g., Atrial Fibrillation).
3.  **Report Generation:** Creating automated, clinically accurate text reports for doctors.

---

## ⚡ Key Features

* **🧠 AI-Powered Core:** Utilizes a 4-bit quantized **Gemma-2B** model with custom LoRA adapters for high efficiency and accuracy.
* **👁️ Multimodal Capabilities:** Handles both visual data (X-Ray images) and time-series data (ECG signals).
* **🛡️ Privacy & Security:** Includes built-in de-identification scripts to scrub patient metadata (compliant with health data regulations).
* **💻 Interactive Dashboard:** A modern, dark-themed web interface built with **Streamlit** for seamless user experience.
* **⚡ Real-Time Inference:** Optimized for fast prediction using GPU acceleration and quantization.

---

## 🛠️ Tech Stack

* **Language:** Python 3.10+
* **Frontend:** Streamlit
* **AI/ML Frameworks:** PyTorch, Transformers (Hugging Face), PEFT (LoRA), BitsAndBytes
* **Data Processing:** Pandas, NumPy, OpenCV, WFDB
* **Deployment:** Ngrok (for tunneling), Google Colab (Environment)

---

## 📂 Datasets Used

The model was trained and validated on standard medical datasets:
* **NIH Chest X-Ray Dataset:** For pneumonia detection and lung opacity analysis.
* **PTB-XL ECG Database:** A large-scale database for electrocardiography analysis.

---

## 🔧 Installation & Usage

To run the RhythmRay dashboard locally or on a cloud environment:

1.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/YazanAlhusseini/RhythmRay.git](https://github.com/YazanAlhusseini/RhythmRay.git)
    cd RhythmRay
    ```

2.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the Application:**
    ```bash
    streamlit run app.py
    ```

---

## 👥 Development Team

This project was brought to life by a dedicated team of Computer Science students:

* **Yazan** 
* **Raad** 
* **Osama** 
* **Khalid** 
* **Thamer** 

---

## 🎓 Acknowledgments

Special thanks to **Umm Al-Qura University** and the **College of Computer and Information Systems** for their support and resources.

---
*© 2026 RhythmRay AI Project. All Rights Reserved.*
