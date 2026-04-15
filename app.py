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

# --- Custom UI Styling (CSS) ---
st.markdown("""
<style>
    [data-testid="stDecoration"], .stDeployButton { display:none; }
    .stButton>button {
        background: linear-gradient(90deg,#00C9FF,#92FE9D);
        color:#004d40; border:none; font-weight:bold;
        height:45px; width:100%; border-radius:10px;
    }
    .result-card {
        background:#1e1e1e; border-radius:12px; padding:20px;
        margin:10px 0; border-left:5px solid #00C9FF; color:white;
    }
</style>
""", unsafe_allow_html=True)

# --- Configuration and Paths ---
# Local paths for public repository compatibility
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

# --- Model Loading Functions (With Caching) ---

@st.cache_resource
def load_cxr_model():
    """Initializes ResNet50 for Chest X-Ray classification."""
    model = tv_models.resnet50(weights=None)
    model.fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(model.fc.in_features, 13))
    if os.path.exists(CXR_PATH):
        ckpt = torch.load(CXR_PATH, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
    return model.to(device).eval()

@st.cache_resource
def load_ecg_model():
    """Initializes EfficientNet-B0 for ECG arrhythmia detection."""
    base = tv_models.efficientnet_b0(weights=None)
    in_f = base.classifier[1].in_features
    base.classifier = nn.Sequential(
        nn.Dropout(0.4), nn.Linear(in_f, 256),
        nn.ReLU(), nn.Dropout(0.2), nn.Linear(256, 5) # 5 cardiac classes
    )
    if os.path.exists(ECG_PATH):
        ckpt = torch.load(ECG_PATH, map_location=device)
        base.load_state_dict(ckpt["model_state_dict"])
    return base.to(device).eval()

@st.cache_resource
def load_gemma_llm(hf_token):
    """Loads MedGemma-2B with 4-bit quantization via BitsAndBytes."""
    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True
    )
    tokenizer = AutoTokenizer.from_pretrained(GEMMA_ID, token=hf_token)
    model = AutoModelForCausalLM.from_pretrained(
        GEMMA_ID, quantization_config=bnb, device_map="auto", token=hf_token
    )
    return tokenizer, model

# --- Inference Logic ---

def predict(image_path, mode):
    """Handles image preprocessing and neural network forward pass."""
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    img = Image.open(image_path).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(device)
    
    # Select expert model based on modality
    model = load_cxr_model() if mode == "CXR" else load_ecg_model()
    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1)[0]
    
    idx = probs.argmax().item()
    label = CXR_LABELS[idx] if mode == "CXR" else ["CLBBB","CRBBB","NORM","PACE","PVC"][idx]
    return label, round(probs[idx].item()*100, 2)

# --- User Interface (UI) Layout ---

with st.sidebar:
    st.title("⚙️ Control Panel")
    hf_token = st.text_input("HuggingFace Token", type="password", help="Required to access MedGemma LLM weights")
    st.divider()
    scan_type = st.radio("Select Diagnostic Modality", ["CXR - Chest X-Ray", "ECG - Electrocardiogram"])
    st.info("Ensure .pth files are located in the /models directory for local inference.")

st.title("🩺 RhythmRay AI: System of Experts")
st.caption("Advanced Multi-Modal Diagnostic Platform for Radiology & Cardiology")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📤 Data Ingestion")
    uploaded_file = st.file_uploader("Upload Medical Scan (CXR or ECG)", type=["jpg", "png", "jpeg"])
    if uploaded_file:
        st.image(uploaded_file, caption="Input Scan Preview", use_container_width=True)

with col2:
    st.subheader("🧠 Cognitive Analysis")
    if uploaded_file and hf_token:
        if st.button("Generate Diagnostic Report"):
            with st.spinner("Analyzing via Vision Experts..."):
                # Save uploaded file to a temporary location for processing
                with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
                    tmp.write(uploaded_file.getvalue())
                    tmp_path = tmp.name
                
                mode = "CXR" if "CXR" in scan_type else "ECG"
                diag, conf = predict(tmp_path, mode)
                
                # Display Results
                st.markdown(f"""
                <div class="result-card">
                    <h4>Primary Diagnosis: {diag}</h4>
                    <p>AI Confidence Score: <b>{conf}%</b></p>
                </div>
                """, unsafe_allow_html=True)
                
                # Clean up temporary file
                os.unlink(tmp_path)
    elif uploaded_file and not hf_token:
        st.warning("HuggingFace Token is required in the sidebar to activate the Cognitive Layer.")
except Exception as e:
    print(f"Error: {e}")
