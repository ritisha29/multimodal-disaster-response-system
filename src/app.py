"""
Streamlit Live Demonstration Application for Multimodal Disaster Tweet Informativeness Classification.

Designed for live faculty demonstration:
1. Reuses MultimodalFusionModel from src/fusion_model.py.
2. Accepts tweet text and uploaded images (JPG/PNG).
3. Compares Multimodal Fusion predictions side-by-side with Text-Only and Image-Only baselines.
4. Performs second-stage Disaster Analysis (Disaster Type & NER Location Extraction).
5. Displays real-world Emergency Response Panel with End-to-End Pipeline Step Flow Indicator, simulated alert dispatch, NDRF relief resources, and Google Places API Nearby Hospitals lookup.
6. Robust error handling for corrupt images or missing inputs.
"""

import os
import sys
import time
import datetime
from typing import Dict, List, Any
import torch
import torch.nn as nn
from PIL import Image
import pandas as pd
import streamlit as st
from torchvision import transforms, models
try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
except ImportError:
    from transformers.models.auto import AutoTokenizer, AutoModelForSequenceClassification

# Ensure src directory is in Python path for clean imports
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

try:
    from fusion_model import MultimodalFusionModel
except ImportError:
    from src.fusion_model import MultimodalFusionModel

try:
    from disaster_analysis import analyze_disaster_info, classify_image_disaster_type
except ImportError:
    from src.disaster_analysis import analyze_disaster_info, classify_image_disaster_type

try:
    from resource_matcher import match_resources
except ImportError:
    from src.resource_matcher import match_resources

try:
    from hospital_finder import find_hospitals_for_location
except ImportError:
    from src.hospital_finder import find_hospitals_for_location


LABEL_NAMES = {0: "Not Informative", 1: "Informative"}


@st.cache_resource
def load_all_models():
    """
    Loads all trained model checkpoints and tokenizer, caching them in memory for fast inference.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    project_root = os.path.dirname(script_dir)
    models_dir = os.path.join(project_root, "models")

    # 0. Load HuggingFace Tokenizer
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    # 1. Multimodal Fusion Model
    fusion_ckpt = os.path.join(models_dir, "fusion_model.pt")
    fusion_model = MultimodalFusionModel(text_model_name="distilbert-base-uncased", dropout_rate=0.3)
    if os.path.exists(fusion_ckpt):
        fusion_model.load_state_dict(torch.load(fusion_ckpt, map_location=device))
    fusion_model.to(device)
    fusion_model.eval()

    # 2. Text-Only Baseline (DistilBERT)
    text_ckpt = os.path.join(models_dir, "text_baseline.pt")
    text_model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)
    if os.path.exists(text_ckpt):
        text_model.load_state_dict(torch.load(text_ckpt, map_location=device))
    text_model.to(device)
    text_model.eval()

    # 3. Image-Only Baseline (ResNet-18)
    image_ckpt = os.path.join(models_dir, "image_baseline.pt")
    try:
        image_model = models.resnet18(weights=None)
    except AttributeError:
        image_model = models.resnet18(pretrained=False)
    image_model.fc = nn.Linear(image_model.fc.in_features, 2)
    if os.path.exists(image_ckpt):
        image_model.load_state_dict(torch.load(image_ckpt, map_location=device))
    image_model.to(device)
    image_model.eval()

    return tokenizer, fusion_model, text_model, image_model, device


def get_image_transforms():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def process_single_image_pipeline(
    raw_image: Image.Image,
    image_name: str,
    caption_text: str,
    tokenizer,
    fusion_model,
    text_model,
    image_model,
    device,
    img_transform
) -> Dict[str, Any]:
    """
    Runs the complete multimodal classification and disaster response pipeline
    strictly independently for a single image + caption pair.
    """
    caption_clean = caption_text.strip() if caption_text else ""

    # 1. Preprocess text (DistilBERT)
    text_input_to_model = caption_clean
    st.write(f"DEBUG - text passed to DistilBERT: {repr(text_input_to_model)}")
    print(f"DEBUG - text passed to DistilBERT: {repr(text_input_to_model)}")

    text_encoding = tokenizer(
        caption_clean,
        padding=True,
        truncation=True,
        max_length=64,
        return_tensors="pt"
    )
    input_ids = text_encoding['input_ids'].to(device)
    attention_mask = text_encoding['attention_mask'].to(device)

    # 2. Preprocess image (ResNet-18)
    img_tensor = img_transform(raw_image).unsqueeze(0).to(device)

    with torch.no_grad():
        # A. Multimodal Fusion Model
        fusion_logits = fusion_model(input_ids=input_ids, attention_mask=attention_mask, images=img_tensor)
        fusion_probs = torch.softmax(fusion_logits, dim=1)[0]
        fusion_pred = int(torch.argmax(fusion_probs).item())
        fusion_conf = fusion_probs[fusion_pred].item() * 100.0

        # B. Text-Only Baseline (DistilBERT)
        text_outputs = text_model(input_ids=input_ids, attention_mask=attention_mask)
        text_probs = torch.softmax(text_outputs.logits, dim=1)[0]
        text_pred = int(torch.argmax(text_probs).item())
        text_conf = text_probs[text_pred].item() * 100.0

        # C. Image-Only Baseline (ResNet-18)
        image_logits = image_model(img_tensor)
        image_probs = torch.softmax(image_logits, dim=1)[0]
        image_pred = int(torch.argmax(image_probs).item())
        image_conf = image_probs[image_pred].item() * 100.0

    # 3. Second-Stage Disaster Analysis (Text NER & Keyword categorization)
    analysis_res = analyze_disaster_info(caption_clean)
    d_type = analysis_res["disaster_type"]
    d_icon = analysis_res["icon"]
    d_loc = analysis_res["location"]

    # 4. Image Disaster Signal (CLIP zero-shot + filename heuristics)
    image_d_type = classify_image_disaster_type(raw_image, filename=image_name)
    is_mismatch = (
        image_d_type != "Unclear"
        and d_type != "Other / General Emergency"
        and d_type != image_d_type
    )

    # 5. Relief Resource Matcher
    matched_res = match_resources(d_loc)

    # 6. OpenStreetMap Nearby Hospitals Lookup
    hosp_result = find_hospitals_for_location(d_loc, radius_km=10.0)

    # 7. Dispatch Timestamp
    timestamp_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S IST")

    return {
        "image_name": image_name,
        "raw_image": raw_image,
        "caption_text": caption_clean,
        "fusion_pred": fusion_pred,
        "fusion_label": LABEL_NAMES[fusion_pred],
        "fusion_conf": fusion_conf,
        "text_pred": text_pred,
        "text_label": LABEL_NAMES[text_pred],
        "text_conf": text_conf,
        "image_pred": image_pred,
        "image_label": LABEL_NAMES[image_pred],
        "image_conf": image_conf,
        "disaster_type": d_type,
        "disaster_icon": d_icon,
        "location": d_loc,
        "image_disaster_type": image_d_type,
        "is_mismatch": is_mismatch,
        "matched_resources": matched_res,
        "hospital_result": hosp_result,
        "timestamp": timestamp_str
    }


def render_detailed_results(res: Dict[str, Any]):
    """
    Renders the full detailed results (classification cards, baselines comparison,
    second-stage disaster analysis, emergency response panel) inside an expander.
    """
    # Preview Image & Caption in 2 columns
    img_col, cap_col = st.columns([1, 2], gap="medium")
    with img_col:
        st.image(res["raw_image"], caption=res["image_name"], use_container_width=True)
    with cap_col:
        st.markdown("**📝 Post Caption / Tweet Text:**")
        caption_show = res["caption_text"] if res["caption_text"] else "*(No caption provided — evaluated using image features and default fallbacks)*"
        st.info(caption_show)

    # 1. Primary Multimodal Fusion Model Prediction (Large Card)
    card_class = "prediction-card-informative" if res["fusion_pred"] == 1 else "prediction-card-not-informative"
    title_class = "pred-title-informative" if res["fusion_pred"] == 1 else "pred-title-not-informative"

    st.markdown(f"""
        <div class="{card_class}">
            <div style="font-size: 1rem; color: #64748b; font-weight: 600; text-transform: uppercase; letter-spacing: 1px;">Multimodal Fusion Model Prediction</div>
            <div class="{title_class}">{res['fusion_label']}</div>
            <div class="confidence-text">Confidence: {res['fusion_conf']:.2f}%</div>
        </div>
    """, unsafe_allow_html=True)

    # 2. Side-by-Side Modality Comparison
    st.markdown("#### 🔬 Side-by-Side Modality Comparison")
    st.caption("Demonstrating how Multimodal Fusion combines textual and visual modalities to outperform individual baselines:")

    comp_col1, comp_col2, comp_col3 = st.columns(3)

    with comp_col1:
        t_badge = "🟢" if res["text_pred"] == 1 else "⚪"
        st.markdown(f"""
            <div class="comparison-box">
                <div style="font-size: 0.9rem; font-weight: 700; color: #475569;">📝 Text-Only (DistilBERT)</div>
                <hr style="margin: 0.5rem 0;">
                <div style="font-size: 1.3rem; font-weight: 800; color: #1e293b;">{t_badge} {res['text_label']}</div>
                <div style="font-size: 0.95rem; color: #64748b; margin-top: 0.3rem;">Confidence: <b>{res['text_conf']:.1f}%</b></div>
            </div>
        """, unsafe_allow_html=True)

    with comp_col2:
        i_badge = "🟢" if res["image_pred"] == 1 else "⚪"
        st.markdown(f"""
            <div class="comparison-box">
                <div style="font-size: 0.9rem; font-weight: 700; color: #475569;">🖼️ Image-Only (ResNet-18)</div>
                <hr style="margin: 0.5rem 0;">
                <div style="font-size: 1.3rem; font-weight: 800; color: #1e293b;">{i_badge} {res['image_label']}</div>
                <div style="font-size: 0.95rem; color: #64748b; margin-top: 0.3rem;">Confidence: <b>{res['image_conf']:.1f}%</b></div>
            </div>
        """, unsafe_allow_html=True)

    with comp_col3:
        f_badge = "🟢" if res["fusion_pred"] == 1 else "⚪"
        st.markdown(f"""
            <div class="comparison-box" style="border: 2px solid #3b82f6; background-color: #eff6ff;">
                <div style="font-size: 0.9rem; font-weight: 700; color: #1d4ed8;">⚡ Multimodal Fusion (Proposed)</div>
                <hr style="margin: 0.5rem 0; border-color: #bfdbfe;">
                <div style="font-size: 1.3rem; font-weight: 800; color: #1e3a8a;">{f_badge} {res['fusion_label']}</div>
                <div style="font-size: 0.95rem; color: #1e40af; margin-top: 0.3rem;">Confidence: <b>{res['fusion_conf']:.1f}%</b></div>
            </div>
        """, unsafe_allow_html=True)

    # 3. Second-Stage Disaster Analysis
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### 📍 Second-Stage Disaster Analysis")
    st.caption("Second-stage intelligence extraction performed for this post:")

    if res["is_mismatch"]:
        st.warning(f"⚠️ **Multimodal Mismatch Detected**: Text suggests **{res['disaster_type']}**, but image suggests **{res['image_disaster_type']}** — this post may need manual review.")

    anal_col1, anal_col2 = st.columns(2)
    with anal_col1:
        st.markdown(f"""
            <div class="analysis-box">
                <div style="font-size: 0.9rem; font-weight: 700; color: #475569;">🏷️ Detected Disaster Type</div>
                <div style="font-size: 1.5rem; font-weight: 800; color: #0f172a; margin-top: 0.4rem;">
                    {res['disaster_icon']} {res['disaster_type']}
                </div>
            </div>
        """, unsafe_allow_html=True)

    with anal_col2:
        st.markdown(f"""
            <div class="analysis-box">
                <div style="font-size: 0.9rem; font-weight: 700; color: #475569;">📍 Extracted Location (NER)</div>
                <div style="font-size: 1.5rem; font-weight: 800; color: #0f172a; margin-top: 0.4rem;">
                    📌 {res['location']}
                </div>
            </div>
        """, unsafe_allow_html=True)

    # 4. Emergency Response Panel
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### 🚨 Emergency Response Panel")
    st.caption("Real-world automated alert dispatch, Indian relief resource routing, and live hospital lookup:")

    # Pipeline Step Flow
    st.markdown(f"""
        <div class="pipeline-flow-container">
            <div class="pipeline-step">
                <div class="step-num">STEP 1</div>
                <div class="step-icon">🤖</div>
                <div class="step-title">Multimodal AI</div>
                <div class="step-desc">{res['fusion_label']}<br><b>({res['fusion_conf']:.1f}% Conf)</b></div>
            </div>
            <div class="pipeline-arrow">➔</div>
            <div class="pipeline-step">
                <div class="step-num">STEP 2</div>
                <div class="step-icon">{res['disaster_icon']}</div>
                <div class="step-title">Disaster Type</div>
                <div class="step-desc"><b>{res['disaster_type']}</b></div>
            </div>
            <div class="pipeline-arrow">➔</div>
            <div class="pipeline-step">
                <div class="step-num">STEP 3</div>
                <div class="step-icon">📍</div>
                <div class="step-title">Location NER</div>
                <div class="step-desc"><b>{res['location']}</b></div>
            </div>
            <div class="pipeline-arrow">➔</div>
            <div class="pipeline-step">
                <div class="step-num">STEP 4</div>
                <div class="step-icon">🏥</div>
                <div class="step-title">Resource Match</div>
                <div class="step-desc"><b>{len(res['matched_resources'])} Resources</b></div>
            </div>
            <div class="pipeline-arrow">➔</div>
            <div class="pipeline-step step-final">
                <div class="step-num">STEP 5</div>
                <div class="step-icon">🚨</div>
                <div class="step-title">Alert Dispatch</div>
                <div class="step-desc"><b>Payload Ready</b><br>(Simulated)</div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # Simulated Alert Banner
    st.markdown("""
        <div class="simulated-alert-banner">
            <div style="font-weight: 800; color: #d46b08; font-size: 1.05rem;">
                ⚠️ SIMULATED ALERT — Prototype demonstration — no real alert is sent.
            </div>
            <div style="color: #595959; font-size: 0.88rem; margin-top: 0.2rem;">
                Mock automated emergency dispatch payload generated from live classifier output.
            </div>
        </div>
    """, unsafe_allow_html=True)

    # Formatted Payload Box
    st.markdown(f"""
        <div class="payload-preview-box">
            <div style="color: #38bdf8; font-weight: bold; font-size: 1.05rem; margin-bottom: 0.5rem;">[EMERGENCY_DISPATCH_PAYLOAD_PREVIEW]</div>
            <div>--------------------------------------------------</div>
            <div><b style="color: #a855f7;">EVENT_TYPE</b>  : {res['disaster_icon']} {res['disaster_type'].upper()}</div>
            <div><b style="color: #a855f7;">LOCATION</b>    : 📍 {res['location']}</div>
            <div><b style="color: #a855f7;">CONFIDENCE</b>  : {res['fusion_conf']:.2f}%</div>
            <div><b style="color: #a855f7;">TIMESTAMP</b>   : {res['timestamp']}</div>
            <div><b style="color: #a855f7;">DISPATCH_PRIORITY</b>: HIGH (INFORMATIONAL DISASTER INCIDENT)</div>
            <div>--------------------------------------------------</div>
        </div>
    """, unsafe_allow_html=True)

    # Nearby Relief Resources Table
    st.markdown("##### 📦 Nearby Relief Resources")
    d_loc = res["location"]
    if "location not specified" in d_loc.lower():
        st.info("ℹ️ Location not specified — showing general national relief resources (NDRF / Red Cross India).")
    else:
        st.success(f"📍 Matched emergency relief resources for detected location **'{d_loc}'**:")

    matched_res = res["matched_resources"]
    if matched_res:
        table_data = []
        for r in matched_res:
            rtype = r.get('resource_type', '').lower()
            icon_type = "🏠 Shelter" if "shelter" in rtype else ("🍱 Food" if "food" in rtype else ("🚰 Water" if "water" in rtype else "📦 General"))
            table_data.append({
                "Type": icon_type,
                "Name": r.get('name', ''),
                "City": r.get('city', ''),
                "State/Region": r.get('region', ''),
                "Capacity": f"{r.get('capacity', 'N/A')} units",
                "Contact Information": r.get('contact_info', '')
            })
        st.dataframe(pd.DataFrame(table_data), use_container_width=True, hide_index=True)

    # Nearby Hospitals Table
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("##### 🏥 Nearby Hospitals (OpenStreetMap / Local Emergency DB)")
    hosp_result = res["hospital_result"]
    if hosp_result["status_code"] in ["OK", "OFFLINE_MODE"]:
        st.success(f"🏥 {hosp_result['message']}")
    elif hosp_result["status_code"] == "UNSPECIFIED_LOCATION":
        st.info("ℹ️ No specific location detected in this post — showing general nearby emergency hospitals instead.")
    else:
        st.info(f"ℹ️ {hosp_result['message']}")

    if hosp_result.get("hospitals"):
        hosp_df_data = []
        for h in hosp_result["hospitals"]:
            hosp_df_data.append({
                "Hospital Name": h["name"],
                "Address": h["address"],
                "Distance": h["distance_km"],
                "Open Status": h["open_status"],
                "Rating": h["rating"]
            })
        st.dataframe(pd.DataFrame(hosp_df_data), use_container_width=True, hide_index=True)


def main():
    # Configure Streamlit page layout and theme
    st.set_page_config(
        page_title="Multimodal Disaster Informativeness Classifier",
        page_icon="🚨",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Custom CSS for high-contrast, presentation-ready styling
    st.markdown("""
        <style>
        .main-header {
            font-size: 2.2rem;
            font-weight: 700;
            color: #1e293b;
            margin-bottom: 0.2rem;
        }
        .sub-header {
            font-size: 1.1rem;
            color: #64748b;
            margin-bottom: 1.5rem;
        }
        .prediction-card-informative {
            background-color: #f0fdf4;
            border: 2px solid #22c55e;
            border-radius: 12px;
            padding: 1.25rem;
            text-align: center;
            margin-bottom: 1.5rem;
        }
        .prediction-card-not-informative {
            background-color: #f8fafc;
            border: 2px solid #94a3b8;
            border-radius: 12px;
            padding: 1.25rem;
            text-align: center;
            margin-bottom: 1.5rem;
        }
        .pred-title-informative {
            font-size: 2rem;
            font-weight: 800;
            color: #15803d;
            margin: 0;
        }
        .pred-title-not-informative {
            font-size: 2rem;
            font-weight: 800;
            color: #475569;
            margin: 0;
        }
        .confidence-text {
            font-size: 1.2rem;
            font-weight: 600;
            color: #334155;
            margin-top: 0.4rem;
        }
        .comparison-box {
            background-color: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 8px;
            padding: 1rem;
            text-align: center;
            box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        }
        .analysis-box {
            background-color: #f8fafc;
            border: 1px solid #cbd5e1;
            border-radius: 10px;
            padding: 1.2rem;
            margin-top: 0.5rem;
        }
        .simulated-alert-banner {
            background-color: #fffbe6;
            border: 2px solid #ffe58f;
            border-left: 6px solid #faad14;
            border-radius: 8px;
            padding: 1rem;
            margin-bottom: 1.2rem;
        }
        .payload-preview-box {
            background-color: #0f172a;
            color: #f8fafc;
            border-radius: 10px;
            padding: 1.2rem;
            font-family: 'Courier New', Courier, monospace;
            font-size: 0.95rem;
            margin-bottom: 1.5rem;
            box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1);
        }
        /* PIPELINE FLOW STYLING */
        .pipeline-flow-container {
            display: flex;
            align-items: center;
            justify-content: space-between;
            background-color: #f8fafc;
            border: 1px solid #e2e8f0;
            border-radius: 12px;
            padding: 1.2rem 1rem;
            margin-bottom: 1.5rem;
            box-shadow: 0 2px 4px rgba(0,0,0,0.04);
        }
        .pipeline-step {
            flex: 1;
            background-color: #ffffff;
            border: 1px solid #cbd5e1;
            border-radius: 8px;
            padding: 0.8rem 0.5rem;
            text-align: center;
        }
        .pipeline-step.step-final {
            background-color: #fffbe6;
            border-color: #ffe58f;
        }
        .step-num {
            font-size: 0.7rem;
            font-weight: 800;
            color: #64748b;
            letter-spacing: 0.5px;
        }
        .step-icon {
            font-size: 1.4rem;
            margin: 0.2rem 0;
        }
        .step-title {
            font-size: 0.85rem;
            font-weight: 700;
            color: #1e293b;
        }
        .step-desc {
            font-size: 0.8rem;
            color: #475569;
            margin-top: 0.2rem;
            line-height: 1.2;
        }
        .pipeline-arrow {
            font-size: 1.2rem;
            color: #94a3b8;
            font-weight: bold;
            padding: 0 0.4rem;
        }
        </style>
    """, unsafe_allow_html=True)

    # --- SIDEBAR: Project Overview & Architecture ---
    st.sidebar.title("📌 Project Overview")
    st.sidebar.markdown("""
    **Task:** Multimodal Disaster Tweet Informativeness Classification  
    **Goal:** Classify social media posts during natural disasters as **Informative** (providing critical situational updates) vs. **Not Informative**.

    ---
    ### 📊 Dataset & Encoders
    - **Dataset:** CrisisMMD v2.0 (Disaster Tweets)
    - **Text Encoder:** DistilBERT (`distilbert-base-uncased`) → 768-dim `[CLS]` token embedding.
    - **Visual Encoder:** ResNet-18 → 512-dim penultimate feature vector.

    ---
    ### 🔗 Multimodal Fusion Architecture
    - **Fusion Strategy:** Late Feature Concatenation ($768 + 512 = 1280$-dim joint representation).
    - **Classifier:** 3-Layer MLP ($1280 \\rightarrow 256 \\rightarrow 64 \\rightarrow 2$) with ReLU non-linearities and Dropout ($p=0.3$).

    ---
    ### 🚨 End-to-End Response Pipeline
    - **Post-Processing:** Disaster classification & spaCy Location NER.
    - **Emergency Dispatch:** Automated alert payload preview & NDRF relief resources.
    - **Live Geolocation:** Google Places API nearby hospital lookup.
    """)

    # Main Header
    st.markdown('<div class="main-header">Multimodal Disaster Informativeness Classifier</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Live Multimodal AI Inference & End-to-End Emergency Response Pipeline</div>', unsafe_allow_html=True)

    # Load models
    with st.spinner("Loading PyTorch models into memory..."):
        tokenizer, fusion_model, text_model, image_model, device = load_all_models()
        img_transform = get_image_transforms()

    # Image File Uploader (Multiple files supported)
    uploaded_files = st.file_uploader(
        "Upload One or Multiple Disaster Images (JPG / PNG / WebP):",
        type=["jpg", "jpeg", "png", "webp"],
        accept_multiple_files=True,
        help="Select one or more disaster image files to classify."
    )

    uploaded_items = []
    if uploaded_files:
        st.markdown("---")
        st.subheader("🖼️ Uploaded Images & Per-Image Editable Captions")
        st.caption("Each image is displayed with an editable text area below. Use preset buttons to auto-fill or type/paste your own custom tweet text at any time:")

        # Thumbnails row at a glance
        st.markdown("**Thumbnails Overview:**")
        thumb_cols = st.columns(len(uploaded_files))
        for t_idx, file in enumerate(uploaded_files):
            with thumb_cols[t_idx]:
                try:
                    t_img = Image.open(file).convert("RGB")
                    st.image(t_img, caption=f"#{t_idx+1}: {file.name}", use_container_width=True)
                except Exception as e:
                    st.error(f"Error loading {file.name}: {e}")

        st.markdown("---")
        # Per-Image Editable Text Area Section
        for idx, file in enumerate(uploaded_files):
            try:
                img = Image.open(file).convert("RGB")
            except Exception as e:
                st.error(f"❌ Error loading '{file.name}': {e}")
                continue

            cap_key = f"caption_text_area_{idx}_{file.name}"
            if cap_key not in st.session_state:
                st.session_state[cap_key] = ""

            img_col, edit_col = st.columns([1, 2.5], gap="large")
            with img_col:
                st.image(img, caption=f"Image {idx+1}: {file.name}", use_container_width=True)
            with edit_col:
                st.markdown(f"#### Image {idx+1}: `{file.name}`")
                st.caption("Auto-fill with preset (or freely edit / type custom text below):")

                p_cols = st.columns(5)
                with p_cols[0]:
                    if st.button("🌊 Kerala", key=f"btn_kerala_{idx}_{file.name}"):
                        st.session_state[cap_key] = "KERALA FLOOD ALERT: Heavy torrential rainfall causing severe flooding in Kochi and Wayanad. NDRF teams deployed. Stay indoors."
                with p_cols[1]:
                    if st.button("🌧️ Assam", key=f"btn_assam_{idx}_{file.name}"):
                        st.session_state[cap_key] = "ASSAM FLOOD UPDATE: Brahmaputra river overflow has inundated several villages near Guwahati. Emergency relief camps operational."
                with p_cols[2]:
                    if st.button("⛰️ Uttarakhand", key=f"btn_uttar_{idx}_{file.name}"):
                        st.session_state[cap_key] = "UTTARAKHAND EMERGENCY: Cloudburst triggered flash flooding near Dehradun. State disaster management alert issued for low areas."
                with p_cols[3]:
                    if st.button("🌪️ Mumbai", key=f"btn_mumbai_{idx}_{file.name}"):
                        st.session_state[cap_key] = "MUMBAI WEATHER WARNING: Severe waterlogging and high tide in Mumbai city. Red alert issued by IMD. Emergency helpline 1916 active."
                with p_cols[4]:
                    if st.button("🧹 Clear", key=f"btn_clear_{idx}_{file.name}"):
                        st.session_state[cap_key] = ""

                # The actual editable st.text_area widget
                caption_val = st.text_area(
                    label=f"Editable Tweet / Caption Text for Image {idx+1}:",
                    key=cap_key,
                    height=105,
                    placeholder="Type or paste any custom tweet text, location, or disaster description here...",
                    help=f"Freely type or edit custom caption text for {file.name}."
                )

                uploaded_items.append({
                    "file": file,
                    "name": file.name,
                    "image": img,
                    "caption": caption_val
                })

            st.markdown("<hr style='margin: 1rem 0; border-color: #e2e8f0;'>", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        classify_all_btn = st.button("🚀 Classify All", type="primary", use_container_width=True)
    else:
        st.info("ℹ️ Upload one or more disaster images above. Once uploaded, each image will have its own dedicated editable text area and auto-fill buttons.")
        classify_all_btn = False

    # --- BATCH MULTI-IMAGE CLASSIFICATION PIPELINE ---
    if classify_all_btn:
        if not uploaded_items:
            st.warning("⚠️ Please upload at least one valid image to classify.")
            return

        st.markdown("---")
        st.subheader("📊 Multi-Image Classification Results")

        # 3. Progress indicator while looping through images
        progress_bar = st.progress(0, text="Initializing multimodal classification pipeline...")
        results = []
        total = len(uploaded_items)

        try:
            for idx, item in enumerate(uploaded_items, start=1):
                progress_bar.progress(
                    (idx - 1) / total,
                    text=f"Processing image {idx} of {total}: {item['name']}..."
                )
                res = process_single_image_pipeline(
                    raw_image=item["image"],
                    image_name=item["name"],
                    caption_text=item["caption"],
                    tokenizer=tokenizer,
                    fusion_model=fusion_model,
                    text_model=text_model,
                    image_model=image_model,
                    device=device,
                    img_transform=img_transform
                )
                results.append(res)
                hospitals = res["hospital_result"].get("hospitals", [])
                st.write(f"DEBUG - Image {idx}: filename={item['name']}, detected_location={res['location']}, hospitals={[h['name'] for h in hospitals]}")

            progress_bar.progress(1.0, text=f"✅ Completed classification for all {total} image{'s' if total > 1 else ''}!")
            time.sleep(0.3)
            progress_bar.empty()

        except Exception as e:
            progress_bar.empty()
            st.error(f"❌ An error occurred during model inference: {e}")
            return

        # 4. SUMMARY TABLE at the top of the results
        st.markdown("### 📋 Batch Summary Overview")
        st.caption("At-a-glance prediction overview across all uploaded images:")

        summary_rows = []
        for r in results:
            badge = "🟢" if r["fusion_pred"] == 1 else "⚪"
            summary_rows.append({
                "Image Name": r["image_name"],
                "Prediction": f"{badge} {r['fusion_label']}",
                "Confidence": f"{r['fusion_conf']:.2f}%",
                "Disaster Type": f"{r['disaster_icon']} {r['disaster_type']}",
                "Location": f"📌 {r['location']}"
            })
        summary_df = pd.DataFrame(summary_rows)
        st.dataframe(summary_df, use_container_width=True, hide_index=True)

        # 5. DETAILED RESULTS in separate st.expander per image
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### 🔍 Detailed Per-Image Analysis & Emergency Response")
        st.caption("Expand any image card below to inspect its individual multimodal baselines, NER extraction, and simulated emergency response:")

        for idx, res in enumerate(results, start=1):
            expander_title = f"Image {idx}: {res['image_name']} — {res['fusion_label']} ({res['fusion_conf']:.1f}% Confidence)"
            with st.expander(expander_title, expanded=(idx == 1)):
                render_detailed_results(res)


if __name__ == "__main__":
    main()
