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
# Class labels (UNCHANGED)
# ============================================================
CLASS_MAP = {
    0: "Healthy Liver",
    1: "Cirrhosis",
    2: "Hepatitis",
    3: "Fibrosis",
    4: "Suspected Liver Disorder"
}

# ============================================================
# Input UI metadata
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
st.caption("Multiclass ML probability analysis (decision support, not diagnosis)")
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

    # Sort probabilities
    class_probs = sorted(
        [(CLASS_MAP[i], probs[i]) for i in range(len(probs))],
        key=lambda x: x[1],
        reverse=True
    )

    top_class, top_prob = class_probs[0]
    second_class, second_prob = class_probs[1]

    st.success(f"### 🧾 Primary Prediction: **{top_class}**")
    st.info(
        f"Top Probability: **{top_prob:.2%}**  \n"
        f"Second Likely Pattern: **{second_class} ({second_prob:.2%})**"
    )

    st.divider()
    st.subheader("📊 Multi-Class Disease Pattern Analysis")

    # ========================================================
    # CHART 1 — FULL 5-CLASS DISTRIBUTION
    # ========================================================
    labels = [c for c, _ in class_probs]
    values = [p for _, p in class_probs]

    fig1 = go.Figure(
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

    fig1.update_layout(
        title="Disease Pattern Similarity (All 5 Classes)",
        yaxis=dict(range=[0, 1], title="Probability"),
        xaxis_title="Disease Pattern",
    )

    st.plotly_chart(fig1, use_container_width=True)

    # ========================================================
    # CHART 2 — TOP-2 DOMINANCE
    # ========================================================
    fig2 = go.Figure(
        data=[
            go.Bar(
                x=[top_class, second_class],
                y=[top_prob, second_prob],
                marker_color=["#6c5ce7", "#00cec9"]
            )
        ]
    )

    fig2.update_layout(
        title="Top-2 Class Dominance",
        yaxis=dict(range=[0, 1], title="Probability"),
    )

    st.plotly_chart(fig2, use_container_width=True)

    # ========================================================
    # CHART 3 — GROUPED PATTERN VIEW
    # ========================================================
    healthy = probs[0]
    acute = probs[1] + probs[2]      # Cirrhosis + Hepatitis
    chronic = probs[3] + probs[4]    # Fibrosis + Suspected

    fig3 = go.Figure(
        data=[
            go.Bar(
                x=["Healthy Pattern", "Acute Pattern", "Chronic Pattern"],
                y=[healthy, acute, chronic],
                marker_color=["#2ecc71", "#e67e22", "#c0392b"]
            )
        ]
    )

    fig3.update_layout(
        title="Grouped Disease Pattern Similarity",
        yaxis=dict(range=[0, 1], title="Combined Probability"),
    )

    st.plotly_chart(fig3, use_container_width=True)

    # ========================================================
    # CLINICAL INTERPRETATION (CORRECT LOGIC)
    # ========================================================
    st.subheader("🩺 Clinical Interpretation (Decision Support)")

    with st.chat_message("assistant"):

        if top_prob >= 0.80:
            st.write(
                f"🔴 **High confidence pattern detected: {top_class}.**\n\n"
                f"The model shows strong alignment with historical cases labeled as "
                f"**{top_class}**, with limited overlap from other classes.\n\n"
                "✅ **Recommendation:** Prompt clinical evaluation aligned with this pattern is advised."
            )

        elif top_prob >= 0.50:
            st.write(
                f"🟠 **Moderate confidence leaning toward {top_class}.**\n\n"
                f"There is notable overlap with **{second_class}**, indicating uncertainty.\n\n"
                "✅ **Recommendation:** Additional diagnostic testing and short-term follow-up are recommended."
            )

        else:
            st.write(
                "🟡 **Diffuse probability distribution detected.**\n\n"
                "The model does not strongly associate the patient with a single disease pattern.\n\n"
                "✅ **Recommendation:** Continued monitoring and repeat testing if symptoms persist."
            )

    st.caption(
        "⚠️ This system performs statistical pattern similarity analysis and does not provide a medical diagnosis."
    )

# ============================================================
# Footer
# ============================================================
st.divider()
st.caption("LightGBM pipeline • Multiclass probability interpretation • Ethical ML decision support")
