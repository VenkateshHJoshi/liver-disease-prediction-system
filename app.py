# ============================================================
# Imports & global config
# ============================================================
import streamlit as st
import pandas as pd
import joblib
import numpy as np
import plotly.graph_objects as go
import warnings
from sklearn import set_config

# Keep pandas through pipeline (feature-name safety)
set_config(transform_output="pandas")
warnings.filterwarnings("ignore")

# ============================================================
# Load trained pipeline
# ============================================================
@st.cache_resource
def load_pipeline():
    return joblib.load("liver_pipeline.joblib")

bundle = load_pipeline()
pipeline = bundle["pipeline"]
feature_names = bundle["feature_names"]

# ============================================================
# Human-readable class labels
# ============================================================
CLASS_MAP = {
    0: "Healthy Liver",
    1: "Cirrhosis",
    2: "Hepatitis",
    3: "Fibrosis",
    4: "Suspected Liver Disorder"
}

# ============================================================
# Medical-grade UI configuration
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
    "gamma glutamyl transferase": {"label": "GGT – Gamma GT (U/L)", "min": 5, "max": 300, "default": 30},
    "protein": {"label": "Total Protein (g/dL)", "min": 4.0, "max": 9.0, "default": 7.2},
}

# ============================================================
# Page config
# ============================================================
st.set_page_config(
    page_title="Liver Health AI",
    page_icon="🩺",
    layout="centered"
)

st.title("🩺 Liver Health AI Dashboard")
st.caption("Clinical inputs → ML pipeline → visual decision support")

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

        # Sex (human-friendly)
        if key in ["sex", "gender"]:
            sex = st.selectbox("Sex", ["Male", "Female"])
            input_data[feature] = 1 if sex == "Male" else 0

        # Numeric features
        else:
            ui = FEATURE_UI.get(key, {})
            input_data[feature] = st.number_input(
                label=ui.get("label", feature.replace("_", " ").title()),
                min_value=ui.get("min", 0.0),
                max_value=ui.get("max", 1000.0),
                value=ui.get("default", 1.0),
                help="Enter raw lab value (no transformation needed)"
            )

# Preserve feature order
input_df = pd.DataFrame([input_data], columns=feature_names)

# ============================================================
# PREDICTION
# ============================================================
st.divider()

if st.button("🔍 Analyze Liver Health"):

    probs = pipeline.predict_proba(input_df)[0]
    pred_class = int(np.argmax(probs))
    confidence = float(probs[pred_class])
    disease = CLASS_MAP[pred_class]

    # Risk bucket
    if confidence < 0.45:
        risk = "Low"
        risk_color = "green"
    elif confidence < 0.75:
        risk = "Medium"
        risk_color = "orange"
    else:
        risk = "High"
        risk_color = "red"

    # ========================================================
    # RESULT SUMMARY
    # ========================================================
    st.success(f"### 🧾 Diagnosis: **{disease}**")
    st.info(f"Confidence: **{confidence:.2%}** | Risk Level: **{risk}**")

    st.divider()
    st.subheader("📊 Visual Health Analysis")

    # ========================================================
    # DASHBOARD (2 × 2)
    # ========================================================
    col1, col2 = st.columns(2)

    # ---- Chart 1: Risk Gauge ----
    with col1:
        fig1 = go.Figure(go.Indicator(
            mode="gauge+number",
            value=confidence * 100,
            number={"suffix": "%"},
            title={"text": "Overall Risk Confidence"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": risk_color},
                "steps": [
                    {"range": [0, 45], "color": "#b6f2c2"},
                    {"range": [45, 75], "color": "#ffeaa7"},
                    {"range": [75, 100], "color": "#fab1a0"},
                ],
            },
        ))
        st.plotly_chart(fig1, use_container_width=True)

    # ---- Chart 2: Top-2 Probabilities ----
    with col2:
        top_idx = np.argsort(probs)[-2:][::-1]
        fig2 = go.Figure(
            data=[
                go.Bar(
                    x=[CLASS_MAP[i] for i in top_idx],
                    y=probs[top_idx],
                    marker_color=["#0984e3", "#6c5ce7"]
                )
            ]
        )
        fig2.update_layout(
            title="Most Likely Conditions",
            yaxis=dict(range=[0, 1], title="Probability"),
        )
        st.plotly_chart(fig2, use_container_width=True)

    col3, col4 = st.columns(2)

    # ---- Chart 3: Feature Profile ----
    with col3:
        values = np.array([float(input_data[f]) for f in feature_names])
        normalized = values / (np.mean(values) + 1e-6)

        fig3 = go.Figure(
            data=[go.Bar(x=list(range(len(normalized))), y=normalized)]
        )
        fig3.add_hline(y=1, line_dash="dash")
        fig3.update_layout(
            title="Overall Feature Profile (Relative Scale)",
            xaxis_title="Feature Index",
            yaxis_title="Relative Magnitude",
        )
        st.plotly_chart(fig3, use_container_width=True)

    # ---- Chart 4: Top Feature Deviations ----
    with col4:
        deviations = np.abs(normalized - 1.0)
        top_k = np.argsort(deviations)[-5:][::-1]

        fig4 = go.Figure(
            data=[
                go.Bar(
                    x=[feature_names[i].replace("_", " ") for i in top_k],
                    y=deviations[top_k],
                    marker_color="#d63031"
                )
            ]
        )
        fig4.update_layout(
            title="Top Influential Feature Deviations",
            yaxis_title="Deviation Strength",
        )
        st.plotly_chart(fig4, use_container_width=True)

    # ========================================================
    # AI INTERPRETATION
    # ========================================================
    st.subheader("💬 AI Interpretation")
    with st.chat_message("assistant"):
        st.write(
            f"The model predicts **{disease}** with **{confidence:.2%} confidence**. "
            f"The risk level is **{risk.lower()}**, based on learned interactions across multiple biomarkers. "
            "This enables early detection even when individual lab values appear normal."
        )

# ============================================================
# FOOTER
# ============================================================
st.divider()
st.caption("Pipeline-based ML • Plotly visuals • Clinical decision-support tool")
