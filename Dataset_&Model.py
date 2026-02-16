!pip install -q -U bitsandbytes
!pip install -q -U peft
!pip install -q -U transformers accelerate
!pip install -q wfdb
!pip install -q kaggle openpyxl huggingface_hub


import os
import glob
import ast
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from huggingface_hub import login


HF_TOKEN = "Hugging_Face_Token"
login(token=HF_TOKEN)

print("🚀 Starting RhythmRay Pipeline...")


from google.colab import files
if not os.path.exists('kaggle.json'):
    print("⚠️ Upload your 'kaggle.json' file now:")
    files.upload()
    !mkdir -p ~/.kaggle
    !cp kaggle.json ~/.kaggle/
    !chmod 600 ~/.kaggle/kaggle.json


print("\n[1/3] Checking Data Status...")

if os.path.exists("train_cxr.csv") and os.path.exists("train_ecg.csv"):
    print("   ✅ Data files found! Skipping processing to save time.")
    df_cxr = pd.read_csv("train_cxr.csv")
    df_ecg = pd.read_csv("train_ecg.csv")
    print(f"   📊 Loaded: {len(df_cxr)} X-Rays, {len(df_ecg)} ECGs.")
else:
    print("   ⚙️ Processing Data from Scratch...")


    if not os.path.exists("./nih_sample"):
        !kaggle datasets download -d nih-chest-xrays/sample --unzip -p ./nih_sample


    all_images = glob.glob("./nih_sample/**/*.png", recursive=True)
    path_dict = {os.path.basename(x): x for x in all_images}

    csv_files = glob.glob("./nih_sample/**/sample_labels.csv", recursive=True)
    if csv_files:
        df_cxr = pd.read_csv(csv_files[0])
        df_cxr['Full_Path'] = df_cxr['Image Index'].map(path_dict)
        df_cxr = df_cxr.dropna(subset=['Full_Path'])

        if not df_cxr.empty:
            train_cxr, _ = train_test_split(df_cxr, test_size=0.2, random_state=42)
            train_cxr.to_csv("train_cxr.csv", index=False)
            print(f"   ✅ CXR Processed: {len(df_cxr)} images.")

    if not os.path.exists("./ptb_xl"):
        !kaggle datasets download -d bjoernjostein/ptbxlphysionet --unzip -p ./ptb_xl

    db_files = glob.glob("./ptb_xl/**/ptbxl_database.csv", recursive=True)
    if db_files:
        root_ecg = os.path.dirname(db_files[0])
        df_ptb = pd.read_csv(db_files[0], index_col='ecg_id')
        df_ptb.scp_codes = df_ptb.scp_codes.apply(lambda x: ast.literal_eval(x))

        scp_path = os.path.join(root_ecg, 'scp_statements.csv')
        if os.path.exists(scp_path):
            agg_df = pd.read_csv(scp_path, index_col=0)
            agg_df = agg_df[agg_df.diagnostic == 1]

            def aggr(y_dic):
                return list(set([agg_df.loc[k].diagnostic_class for k in y_dic if k in agg_df.index]))

            df_ptb['label'] = df_ptb.scp_codes.apply(aggr).apply(lambda x: x[0] if len(x)==1 else None)
            df_ptb = df_ptb[df_ptb['label'].isin(['NORM', 'MI', 'STTC', 'CD', 'HYP'])]
            df_ptb['Full_Path'] = df_ptb['filename_lr'].apply(lambda x: os.path.join(root_ecg, x))

            if not df_ptb.empty:
                train_ecg, _ = train_test_split(df_ptb, test_size=0.2, random_state=42)
                train_ecg.to_csv("train_ecg.csv")
                print(f"   ✅ ECG Processed: {len(df_ptb)} records.")


print("\n[3/3] Loading AI Model (MedGemma-2B)...")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

MODEL_ID = "google/gemma-2b-it"

try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=HF_TOKEN)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        token=HF_TOKEN
    )


    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, lora_config)

    print("\n🎉 SUCCESS: Model Loaded & Ready for Training!")
    model.print_trainable_parameters()

except Exception as e:
    print(f"\n❌ Error: {e}")
