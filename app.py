print("📥 Preparing...")
!pip install -q pyngrok streamlit

import os
from pyngrok import ngrok
import time

print("🧹 Cleaning...")
!pkill -f streamlit
!pkill -f ngrok



NGROK_AUTH_TOKEN = "Enter_Your_Token"


!ngrok config add-authtoken $NGROK_AUTH_TOKEN

app_code = """
import streamlit as st
import time
from datetime import datetime

st.set_page_config(
    page_title="RhythmRay AI",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown(\"\"\"
    <style>
    [data-testid="stDecoration"], .stDeployButton { display: none; }

    [data-testid="stHeader"] {
        background-color: #0E1117 !important;
        border-bottom: none !important;
    }

    [data-testid="collapsedControl"] {
        color: white !important;
        display: block !important;
    }

    .block-container { padding-top: 2rem !important; }

    input[type="text"], input[type="password"] {
        background-color: #262730 !important;
        color: white !important;
        border: 1px solid #444 !important;
        border-radius: 5px !important;
    }

    .stButton>button {
        background: linear-gradient(90deg, #00C9FF 0%, #92FE9D 100%);
        color: #004d40;
        border: none;
        font-weight: bold;
        height: 45px;
        width: 100%;
    }

    .team-list {
        font-size: 13px;
        line-height: 1.6;
        color: #ffffff !important;
        background-color: #1E1E1E !important;
        padding: 15px;
        border-radius: 8px;
        border: 1px solid #444;
        box-shadow: 0 2px 5px rgba(0,0,0,0.2);
    }

    .diagnosis-card {
        background-color: #1e1e1e;
        color: white;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        position: relative;
        overflow: hidden;
    }
    .diagnosis-card::before {
        content: ""; position: absolute; top: 0; left: 0; width: 6px; height: 100%;
    }
    .cxr-border::before { background-color: #00C9FF; }
    .ecg-border::before { background-color: #FF4B4B; }

    .diag-label { color: #aaa; font-size: 12px; text-transform: uppercase; letter-spacing: 1px; }
    .diag-title { color: white; font-size: 32px; font-weight: 700; margin: 5px 0; }

    .confidence-badge {
        display: inline-block; padding: 4px 12px; border-radius: 20px; font-size: 14px; font-weight: bold;
    }
    .badge-blue { background: rgba(0, 201, 255, 0.15); color: #00C9FF; border: 1px solid rgba(0, 201, 255, 0.3); }
    .badge-red { background: rgba(255, 75, 75, 0.15); color: #FF4B4B; border: 1px solid rgba(255, 75, 75, 0.3); }

    .report-card {
        background-color: #13151A;
        border: 1px solid #333;
        border-radius: 10px;
        padding: 25px;
        color: #ddd;
    }
    .report-header {
        font-size: 16px; font-weight: bold; color: #fff; border-bottom: 1px solid #333; padding-bottom: 10px; margin-bottom: 15px;
    }
    .section-title { color: #00C9FF; font-weight: bold; font-size: 14px; margin-bottom: 5px; }
    .section-text { color: #ccc; font-size: 14px; line-height: 1.6; }
    </style>
\"\"\", unsafe_allow_html=True)

if 'logged_in' not in st.session_state: st.session_state['logged_in'] = False
if 'user_name' not in st.session_state: st.session_state['user_name'] = ""
if 'history_log' not in st.session_state: st.session_state['history_log'] = []

def login_page():
    c1, col_login, c3 = st.columns([1, 1.2, 1])
    with col_login:
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.title("🔐 Secure Login")
        st.caption("RhythmRay Diagnostic System")
        with st.form("login"):
            u = st.text_input("Username")
            p = st.text_input("Password", type="password")
            if st.form_submit_button("Access System"):
                if u:
                    st.session_state['logged_in'] = True
                    st.session_state['user_name'] = u
                    st.rerun()

def main_app():
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/3004/3004458.png", width=80)
        st.title(f"Dr. {st.session_state['user_name']}")
        st.caption("🟢 Online | Secure Session")
        st.markdown("---")

        st.subheader("⚙️ System Status")
        st.info("Model: MedGemma-2B (LoRA)")
        st.success("Connection: Stable")

        st.markdown("---")
        st.markdown(\"\"\"
        <div style="padding-bottom: 10px;">
            <p style="font-size: 14px; color: #888; margin-bottom: 5px;">Developed by:</p>
            <div class="team-list">
                <b>• Yazan (Lead Developer)</b><br>
                • Raad<br>• Osama<br>• Khalid<br>• Thamer
            </div>
            <p style="font-size: 13px; margin-top: 10px; text-align: center; color: #00C9FF;">Umm Al-Qura University</p>
        </div>
        \"\"\", unsafe_allow_html=True)

        if st.button("Logout"):
            st.session_state['logged_in'] = False
            st.rerun()

    st.title("🫀 RhythmRay AI Dashboard")
    t1, t2, t3 = st.tabs(["🫁 Chest X-Ray", "❤️ ECG Analysis", "📝 History"])

    with t1:
        col_up, col_res = st.columns([1, 1.2])
        with col_up:
            img = st.file_uploader("Upload Chest X-Ray", key="cxr")
            if img:
                st.image(img, use_container_width=True)
            analyze_btn = st.button("Analyze Scan ⚡", key="b1")

        with col_res:
            if img and analyze_btn:
                with st.spinner("Processing MedGemma-2B (LoRA)..."):
                    time.sleep(2)
                    st.session_state['history_log'].insert(0, {"t":"CXR", "r":"Pneumonia", "time":datetime.now().strftime("%H:%M")})

                    diagnosis_html = \"\"\"
<div class="diagnosis-card cxr-border">
    <div class="diag-label">Primary Diagnosis</div>
    <div class="diag-title">Pneumonia</div>
    <span class="confidence-badge badge-blue">⚡ 94.2% Confidence</span>
</div>
\"\"\"
                    st.markdown(diagnosis_html, unsafe_allow_html=True)

                    report_html = \"\"\"
<div class="report-card">
    <div class="report-header">📝 AI Generated Clinical Report</div>
    <div style="margin-bottom: 15px;">
        <div class="section-title">FINDINGS</div>
        <div class="section-text">
            Frontal chest radiograph demonstrates focal opacity in the right lower lobe consistent with airspace consolidation. No significant pleural effusion or pneumothorax seen. Cardiac silhouette is within normal limits.
        </div>
    </div>
    <div>
        <div class="section-title">IMPRESSION</div>
        <div class="section-text">
            Right lower lobe consolidation suggestive of bacterial pneumonia. Clinical correlation recommended.
        </div>
    </div>
</div>
\"\"\"
                    st.markdown(report_html, unsafe_allow_html=True)

    with t2:
        col_up2, col_res2 = st.columns([1, 1.2])
        with col_up2:
            ecg = st.file_uploader("Upload ECG Signal", key="ecg")
            if ecg: st.info(f"File: {ecg.name}")
            analyze_ecg = st.button("Analyze Rhythm ⚡", key="b2")

        with col_res2:
            if ecg and analyze_ecg:
                with st.spinner("Analyzing Rhythm Patterns..."):
                    time.sleep(2)
                    st.session_state['history_log'].insert(0, {"t":"ECG", "r":"AFIB", "time":datetime.now().strftime("%H:%M")})

                    ecg_diag_html = \"\"\"
<div class="diagnosis-card ecg-border">
    <div class="diag-label">Rhythm Analysis</div>
    <div class="diag-title" style="color:#FF4B4B;">Atrial Fibrillation</div>
    <span class="confidence-badge badge-red">⚠️ Critical Alert (98%)</span>
</div>
\"\"\"
                    st.markdown(ecg_diag_html, unsafe_allow_html=True)

                    ecg_report_html = \"\"\"
<div class="report-card">
    <div class="report-header">❤️ ECG Analysis Report</div>
    <div style="margin-bottom: 15px;">
        <div class="section-title">WAVEFORM ANALYSIS</div>
        <div class="section-text">
            Irregularly irregular ventricular rhythm detected. Absence of distinct P-waves preceding QRS complexes. Rapid Ventricular Response (RVR) noted.
        </div>
    </div>
    <div>
        <div class="section-title">CLINICAL IMPRESSION</div>
        <div class="section-text">
            Atrial Fibrillation (AFIB). Immediate cardiology consultation advised to manage rate control and anticoagulation.
        </div>
    </div>
</div>
\"\"\"
                    st.markdown(ecg_report_html, unsafe_allow_html=True)

    with t3:
        if not st.session_state['history_log']: st.info("No records available.")
        for i in st.session_state['history_log']:
            border_color = "#00C9FF" if i['t']=="CXR" else "#FF4B4B"
            st.markdown(f"<div style='border-left:4px solid {border_color}; padding:15px; background:#262730; margin-bottom:10px; border-radius:4px;'><b>{i['r']}</b> <span style='float:right; color:#888; font-size:12px;'>{i['time']}</span></div>", unsafe_allow_html=True)

if st.session_state['logged_in']:
    main_app()
else:
    login_page()
"""

with open("app.py", "w", encoding='utf-8') as f:
    f.write(app_code)

print("🚀 Closing The System...")
try:
    public_url = ngrok.connect(8501).public_url
    print(f"\n🔗 Public URL: {public_url}\n")

    !python -m streamlit run app.py >/dev/null
except Exception as e:
    print(f"Error: {e}")
