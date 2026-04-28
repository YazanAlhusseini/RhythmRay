%%writefile app.py
import streamlit as st
import torch
import torch.nn as nn
from torchvision import models as tv_models, transforms
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from PIL import Image
import tempfile
import os

# --- Page Configuration ---
st.set_page_config(page_title="RhythmRay AI", page_icon="🩺", layout="wide")

# --- Custom Styling (CSS) ---
st.markdown("""
<style>
    [data-testid="stDecoration"], .stDeployButton { display:none; }

    /* Analyze Image Button Style */
    .stButton>button {
        background: linear-gradient(90deg, #00C9FF, #92FE9D);
        color: #004d40;
        border: none;
        font-weight: bold;
        border-radius: 8px;
        padding: 10px 20px;
    }

    /* Main Diagnosis Card */
    .diag-box {
        background-color: #1e1e1e;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 15px;
        border-left: 5px solid #00C9FF;
        color: white;
    }
    .diag-box h2 { margin-top: 0; color: white; }
</style>
""", unsafe_allow_html=True)

# --- Configuration & Security ---
# 🔴 ضع التوكن السري الخاص بك هنا بين علامتي التنصيص 🔴
HF_TOKEN = "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"

MODEL_DIR = "./models"
CXR_PATH = os.path.join(MODEL_DIR, "CXR_ResNet50_v2.pth")
ECG_PATH = os.path.join(MODEL_DIR, "ECG_EffNetB0_V4.pth")
GEMMA_ID = "google/gemma-2b-it"
device = "cuda" if torch.cuda.is_available() else "cpu"

CXR_LABELS = [
    "No Finding","Infiltration","Atelectasis","Effusion",
    "Nodule","Mass","Pneumothorax","Consolidation",
    "Pleural_Thickening","Cardiomegaly","Emphysema","Fibrosis","Edema"
]

# --- Model Loading Functions ---
@st.cache_resource
def load_cxr_model():
    model = tv_models.resnet50(weights=None)
    model.fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(model.fc.in_features, 13))
    if os.path.exists(CXR_PATH):
        ckpt = torch.load(CXR_PATH, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
    return model.to(device).eval()

@st.cache_resource
def load_ecg_model():
    base = tv_models.efficientnet_b0(weights=None)
    in_f = base.classifier[1].in_features
    base.classifier = nn.Sequential(
        nn.Dropout(0.4), nn.Linear(in_f, 256),
        nn.ReLU(), nn.Dropout(0.2), nn.Linear(256, 5)
    )
    if os.path.exists(ECG_PATH):
        ckpt = torch.load(ECG_PATH, map_location=device)
        base.load_state_dict(ckpt["model_state_dict"])
    return base.to(device).eval()

@st.cache_resource
def load_gemma_llm():
    # استخدام التوكن المدمج مباشرة بدون طلبه من الواجهة
    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True
    )
    tokenizer = AutoTokenizer.from_pretrained(GEMMA_ID, token=HF_TOKEN)
    model = AutoModelForCausalLM.from_pretrained(
        GEMMA_ID, quantization_config=bnb, device_map="auto", token=HF_TOKEN
    )
    return tokenizer, model

# --- Inference Logic ---
def predict_vision(image_path, mode):
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    img = Image.open(image_path).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(device)

    model = load_cxr_model() if mode == "CXR" else load_ecg_model()
    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1)[0]

    labels_list = CXR_LABELS if mode == "CXR" else ["CLBBB","CRBBB","NORM","PACE","PVC"]

    all_probs = {labels_list[i]: float(probs[i]) * 100 for i in range(len(labels_list))}
    idx = probs.argmax().item()
    label = labels_list[idx]
    confidence = round(float(probs[idx]) * 100, 2)

    return label, confidence, all_probs

def generate_clinical_report(tokenizer, model, diagnosis, confidence, mode):
    prompt = f"Act as a professional cardiologist/radiologist. Based on the {mode} analysis, the AI detected '{diagnosis}' with a confidence of {confidence}%. Write a concise, 1-2 sentence clinical impression report."

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=100, temperature=0.3)

    report = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return report.replace(prompt, "").strip()

# --- User Interface (UI) Layout ---
with st.sidebar:
    # تم إزالة الصورة من هنا بناءً على طلبك
    st.title("⚙️ Control Panel")

    scan_type = st.radio("Select Diagnostic Modality", ["CXR - Chest X-Ray", "ECG - Electrocardiogram"])

    st.divider()
    st.markdown("### 👨‍💻 Team Members")
    st.markdown("""
    * **Yazan Alhusseini**
    * Raad Aladli
    * Osama Alharbi
    * Thamer Alzahrani
    * Khaled Alsolami
    """)

st.title("🩺 RhythmRay AI: System of Experts")

col1, col2 = st.columns([1, 1.2])

with col1:
    uploaded_file = st.file_uploader("Upload X-Ray or ECG Image", type=["jpg", "png", "jpeg"])
    if uploaded_file:
        st.image(uploaded_file, use_container_width=True)

with col2:
    if uploaded_file:
        if st.button("Analyze Image"):
            if HF_TOKEN == "ضع_التوكن_الخاص_بك_هنا":
                st.error("⚠️ لم تقم بوضع التوكن السري في كود app.py!")
            else:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
                    tmp.write(uploaded_file.getvalue())
                    tmp_path = tmp.name

                mode = "CXR" if "CXR" in scan_type else "ECG"

                with st.spinner("Analyzing image..."):
                    diag, conf, all_probs = predict_vision(tmp_path, mode)

                    st.markdown(f"""
                    <div class="diag-box">
                        <h2>Diagnosis: {diag}</h2>
                        <p>Confidence: {conf}%</p>
                    </div>
                    """, unsafe_allow_html=True)

                    with st.expander("All Probabilities", expanded=True):
                        sorted_probs = sorted(all_probs.items(), key=lambda x: x[1], reverse=True)
                        for cls_name, prob_val in sorted_probs:
                            st.write(f"{cls_name}: {prob_val:.1f}%")
                            st.progress(prob_val / 100.0)

                st.markdown("### Clinical Report")
                with st.spinner("Generating report..."):
                    tokenizer, llm_model = load_gemma_llm()
                    clinical_report = generate_clinical_report(tokenizer, llm_model, diag, conf, mode)
                    st.info(clinical_report)

                    st.download_button(
                        label="📄 Download Report",
                        data=f"Diagnosis: {diag}\nConfidence: {conf}%\n\nReport:\n{clinical_report}",
                        file_name="RhythmRay_Report.txt",
                        mime="text/plain"
                    )

                os.unlink(tmp_path)
