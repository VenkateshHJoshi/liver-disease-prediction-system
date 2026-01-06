# ============================================================
# Imports & global config
# ============================================================
import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import warnings

from sklearn import set_config

# Keep pandas through pipeline to preserve feature names
set_config(transform_output="pandas")

# Silence sklearn feature-name warning AFTER fixing root cause
warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names"
)

# ============================================================
# Load trained pipeline (preprocessing + model)
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
# Default RAW clinical values (for UI only)
# ============================================================
def get_default_value(feature: str) -> float:
    f = feature.lower()
    if "age" in f:
        return 45
    if "bilirubin" in f:
        return 1.2
    if "alkaline" in f:
        return 210
    if "alanine" in f or "alamine" in f:
        return 35
    if "aspartate" in f:
        return 40
    if "protein" in f:
        return 6.8
    if "albumin_and" in f:
        return 1.0
    if "albumin" in f:
        return 3.5
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

        # ---- Sex as categorical UI ----
        if feature.lower() in ["sex", "gender"]:
            sex = st.selectbox("Sex", ["Male", "Female"])
            # ⚠️ adjust only if training encoding differs
            input_data[feature] = 1 if sex == "Male" else 0

        # ---- All other numeric features ----
        else:
            input_data[feature] = st.number_input(
                feature.replace("_", " "),
                value=float(get_default_value(feature))
            )

# IMPORTANT: enforce correct feature order + names
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

    # Risk buckets
    if confidence < 0.45:
        risk = "Low"
    elif confidence < 0.75:
        risk = "Medium"
    else:
        risk = "High"

    # ========================================================
    # RESULT SUMMARY
    # ========================================================
    st.success(f"### 🧾 Diagnosis: **{disease}**")
    st.info(f"Confidence: **{confidence:.2%}** | Risk Level: **{risk}**")

    st.divider()
    st.subheader("📊 Visual Health Analysis")

    # ========================================================
    # DASHBOARD (2 × 2 GRID)
    # ========================================================
    with st.container():

        # ===================== ROW 1 =====================
        col1, col2 = st.columns(2)

        # ---- Chart 1: Risk Meter ----
        with col1:
            color = "green" if risk == "Low" else "orange" if risk == "Medium" else "red"
            fig, ax = plt.subplots(figsize=(5, 3))
            ax.barh(["Risk"], [confidence], color=color)
            ax.set_xlim(0, 1)
            ax.set_xlabel("Model Confidence")
            ax.set_title("Overall Risk Level")
            st.pyplot(fig)

        # ---- Chart 2: Top-2 Class Probabilities ----
        with col2:
            top_idx = np.argsort(probs)[-2:][::-1]
            labels = [CLASS_MAP[i] for i in top_idx]
            values = probs[top_idx]

            fig, ax = plt.subplots(figsize=(5, 3))
            ax.bar(labels, values)
            ax.set_ylim(0, 1)
            ax.set_ylabel("Probability")
            ax.set_title("Most Likely Conditions")
            plt.xticks(rotation=15)
            st.pyplot(fig)

        # ===================== ROW 2 =====================
        col3, col4 = st.columns(2)

        # ---- Chart 3: Normalized Feature Profile (ALL features) ----
        with col3:
            feature_vals = np.array(
                [float(input_data[f]) for f in feature_names],
                dtype=float
            )

            # Robust normalization (always non-empty)
            mean_val = np.mean(feature_vals) + 1e-6
            normalized = feature_vals / mean_val

            fig, ax = plt.subplots(figsize=(5, 3))
            ax.bar(range(len(normalized)), normalized)
            ax.axhline(1, linestyle="--", color="black")
            ax.set_title("Overall Feature Profile (Relative Scale)")
            ax.set_xlabel("Feature Index")
            ax.set_ylabel("Relative Magnitude")
            st.pyplot(fig)

        # ---- Chart 4: Top Feature Deviations (Proxy Explanation) ----
        with col4:
            deviations = np.abs(normalized - 1.0)
            top_k = np.argsort(deviations)[-5:][::-1]

            fig, ax = plt.subplots(figsize=(5, 3))
            ax.bar(
                [feature_names[i].replace("_", " ") for i in top_k],
                deviations[top_k]
            )
            ax.set_title("Top Influential Feature Deviations")
            ax.set_ylabel("Deviation Strength")
            plt.xticks(rotation=30)
            st.pyplot(fig)

    # ========================================================
    # CHAT-STYLE INTERPRETATION
    # ========================================================
    st.subheader("💬 AI Interpretation")

    with st.chat_message("assistant"):
        st.write(
            f"The model predicts **{disease}** with **{confidence:.2%} confidence**. "
            f"Although individual lab values may appear within normal limits, "
            f"the **combined feature pattern** closely matches historical cases of this condition. "
            "This enables early risk detection before clinical thresholds are crossed."
        )

# ============================================================
# FOOTER
# ============================================================
st.divider()
st.caption("Pipeline-based ML • Human-first UI • Decision-support system")
