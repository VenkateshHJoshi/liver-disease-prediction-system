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

# Keep pandas through pipeline
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
# Default RAW clinical values (UI only)
# ============================================================
def get_default_value(feature):
    f = feature.lower()
    if "age" in f: return 45
    if "bilirubin" in f: return 1.2
    if "alkaline" in f: return 210
    if "alanine" in f or "alamine" in f: return 35
    if "aspartate" in f: return 40
    if "protein" in f: return 6.8
    if "albumin_and" in f: return 1.0
    if "albumin" in f: return 3.5
    return 1.0

# ============================================================
# Page config
# ============================================================
st.set_page_config(
    page_title="Liver Health AI",
    page_icon="🩺",
    layout="centered"
)

st.title("🩺 Liver Health AI Dashboard")
st.caption("Raw clinical data → ML pipeline → visual decision support")

st.divider()

# ============================================================
# INPUT SECTION
# ============================================================
st.subheader("📋 Patient Clinical Inputs")

input_data = {}
cols = st.columns(2)

for i, feature in enumerate(feature_names):
    with cols[i % 2]:
        if feature.lower() in ["sex", "gender"]:
            sex = st.selectbox("Sex", ["Male", "Female"])
            input_data[feature] = 1 if sex == "Male" else 0
        else:
            input_data[feature] = st.number_input(
                feature.replace("_", " "),
                value=float(get_default_value(feature))
            )

# Enforce correct order & names
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
    with st.container():

        col1, col2 = st.columns(2)

        # ---------------- Chart 1: Risk Gauge ----------------
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

        # ---------------- Chart 2: Top-2 Probabilities ----------------
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

        # ---------------- Chart 3: Feature Profile ----------------
        with col3:
            values = np.array([float(input_data[f]) for f in feature_names])
            normalized = values / (np.mean(values) + 1e-6)

            fig3 = go.Figure(
                data=[
                    go.Bar(
                        x=list(range(len(normalized))),
                        y=normalized,
                        marker_color="#00b894"
                    )
                ]
            )
            fig3.add_hline(y=1, line_dash="dash", line_color="black")
            fig3.update_layout(
                title="Overall Feature Profile (Relative Scale)",
                xaxis_title="Feature Index",
                yaxis_title="Relative Magnitude",
            )
            st.plotly_chart(fig3, use_container_width=True)

        # ---------------- Chart 4: Top Feature Deviations ----------------
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
            f"The risk level is **{risk.lower()}**, based on learned interactions across multiple features. "
            "This enables early detection even when individual lab values appear normal."
        )

# ============================================================
# FOOTER
# ============================================================
st.divider()
st.caption("Pipeline-based ML • Plotly-powered visuals • Decision-support system")
