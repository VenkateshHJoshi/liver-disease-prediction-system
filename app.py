# ============================================================
# Imports & configuration
# ============================================================
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
from sklearn import set_config
import warnings

set_config(transform_output="pandas")
warnings.filterwarnings("ignore")

# ============================================================
# Load trained pipeline bundle
# ============================================================
@st.cache_resource
def load_pipeline():
    return joblib.load("liver_pipeline.joblib")

bundle = load_pipeline()
pipeline = bundle["pipeline"]
feature_names = bundle["feature_names"]

# ============================================================
# Class names (UNCHANGED – same as training)
# ============================================================
CLASS_MAP = {
    0: "Healthy Liver",
    1: "Cirrhosis",
    2: "Hepatitis",
    3: "Fibrosis",
    4: "Suspected Liver Disorder"
}

# ============================================================
# UI metadata for inputs
# ============================================================
FEATURE_UI = {
    "age": {"label": "Age (years)", "min": 1, "max": 100, "default": 32},
    "albumin": {"label": "Albumin (g/dL)", "min": 1.5, "max": 6.0, "default": 4.5},
    "alkaline phosphatase": {"label": "Alkaline Phosphatase (U/L)", "min": 40, "max": 400, "default": 95},
    "alanine aminotransferase": {"label": "ALT – Alanine Aminotransferase (U/L)", "min": 5, "max": 300, "default": 22},
    "aspartate aminotransferase": {"label": "AST – Aspartate Aminotransferase (U/L)", "min": 5, "max": 300, "default": 24},
    "bilirubin": {"label": "Total Bilirubin (mg/dL)", "min": 0.1, "max": 10.0, "default": 0.8},
    "cholinesterase": {"label": "Cholinesterase (U/L)", "min": 2000, "max": 12000, "default": 7000},
    "cholesterol": {"label": "Cholesterol (mg/dL)", "min": 80, "max": 400, "default": 170},
    "creatinina": {"label": "Creatinine (mg/dL)", "min": 0.3, "max": 5.0, "default": 0.9},
    "gamma glutamyl transferase": {"label": "Gamma GT (U/L)", "min": 5, "max": 300, "default": 30},
    "protein": {"label": "Total Protein (g/dL)", "min": 4.0, "max": 9.0, "default": 7.2},
}

# ============================================================
# Page setup
# ============================================================
st.set_page_config(
    page_title="Liver Disease Pattern Analysis",
    page_icon="🩺",
    layout="centered"
)

st.title("🩺 Liver Disease Pattern Analysis System")
st.caption("ML-based pattern similarity & risk leaning (not a diagnosis)")
st.divider()

# ============================================================
# INPUT SECTION
# ============================================================
st.subheader("📋 Patient Clinical Inputs")

input_data = {}
cols = st.columns(2)

for i, feature in enumerate(feature_names):
    key = feature.lower()

    with cols[i % 2]:
        if key in ["sex", "gender"]:
            sex = st.selectbox("Sex", ["Male", "Female"])
            input_data[feature] = 1 if sex == "Male" else 0
        else:
            ui = FEATURE_UI.get(key, {})
            input_data[feature] = st.number_input(
                label=ui.get("label", feature.replace("_", " ").title()),
                min_value=ui.get("min", 0.0),
                max_value=ui.get("max", 1000.0),
                value=ui.get("default", 1.0),
            )

input_df = pd.DataFrame([input_data], columns=feature_names)

# ============================================================
# ANALYSIS
# ============================================================
st.divider()

if st.button("🔍 Analyze Pattern"):

    probs = pipeline.predict_proba(input_df)[0]
    pred_class = int(np.argmax(probs))
    confidence = float(probs[pred_class])

    st.success(f"### 🧾 Primary Model Output: **{CLASS_MAP[pred_class]}**")
    st.info(f"Model Confidence: **{confidence:.2%}**")

    st.divider()
    st.subheader("📊 Disease Pattern Leaning")

    # ========================================================
    # 1️⃣ Disease Leaning Chart
    # ========================================================
    focus = {
        "Healthy Liver": 0,
        "Cirrhosis": 1,
        "Hepatitis": 2,
        "Fibrosis": 3,
        "Suspected Liver Disorder": 4,
    }

    labels = list(focus.keys())
    values = [probs[focus[l]] for l in labels]

    fig = go.Figure(
        data=[
            go.Bar(
                x=labels,
                y=values,
                marker_color=[
                    "#2ecc71",  # Healthy
                    "#e74c3c",  # Cirrhosis
                    "#f39c12",  # Hepatitis
                    "#d35400",  # Fibrosis
                    "#9b59b6",  # Suspected
                ]
            )
        ]
    )

    fig.update_layout(
        title="Disease Pattern Similarity (Probability Distribution)",
        yaxis=dict(range=[0, 1], title="Probability"),
        xaxis_title="Disease Pattern",
    )

    st.plotly_chart(fig, use_container_width=True)

    # ========================================================
    # 2️⃣ Lean Strength toward Fibrosis / Suspected
    # ========================================================
    fibrosis_prob = probs[3]
    suspected_prob = probs[4]
    lean_strength = fibrosis_prob + suspected_prob

    st.subheader("📌 Chronic Pattern Lean Strength")

    st.metric(
        label="Fibrosis + Suspected Similarity",
        value=f"{lean_strength:.2%}"
    )

    # ========================================================
    # 3️⃣ Recommendation Engine (Probability-based)
    # ========================================================
    st.subheader("🩺 Clinical Recommendation (Decision Support)")

    with st.chat_message("assistant"):

        if lean_strength >= 0.80:
            st.write(
                "🔴 **Strong leaning toward chronic liver disease patterns detected.**\n\n"
                "The patient's biomarker profile closely resembles historical patterns "
                "associated with **fibrosis or early chronic liver disorders**.\n\n"
                "✅ **Recommendation:** Prompt clinical evaluation is advised, including "
                "advanced liver assessment (e.g., FibroScan, imaging, specialist referral)."
            )

        elif lean_strength >= 0.50:
            st.write(
                "🟠 **Moderate leaning toward fibrotic or suspected patterns.**\n\n"
                "Some biomarkers show early deviations commonly seen in progressive liver conditions.\n\n"
                "✅ **Recommendation:** Regular monitoring, repeat liver function tests, "
                "and lifestyle risk assessment are recommended."
            )

        else:
            st.write(
                "🟢 **Low similarity to fibrotic or suspected disease patterns.**\n\n"
                "The current biomarker profile does not strongly align with chronic liver disease patterns.\n\n"
                "✅ **Recommendation:** Continue routine health check-ups and preventive care."
            )

    # ========================================================
    # Transparency note
    # ========================================================
    st.caption(
        "⚠️ This system performs statistical pattern analysis based on historical data. "
        "It does not provide a medical diagnosis and should be used only as decision-support."
    )

# ============================================================
# Footer
# ============================================================
st.divider()
st.caption("LightGBM Pipeline • Probability-driven interpretation • Ethical ML design")
