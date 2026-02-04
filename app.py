import streamlit as st
import pandas as pd
import joblib
import numpy as np
import re
from pathlib import Path


# App Configuration

st.set_page_config(
    page_title="AutoIntel",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# Constants including confidence of AI

CONFIDENCE_THRESHOLD = 0.20  

MODEL_PATH = Path("models/problem_classifier.joblib")
PARTS_PATH = Path("data/parts.csv")


# Helpers

def looks_like_gibberish(text: str) -> bool:
    """Heuristic checks to reject low-quality or nonsense input."""
    t = text.strip().lower()

    if len(t) < 10:
        return True

    letters = sum(ch.isalpha() for ch in t)
    if letters < 6:
        return True

    non_letters = sum(not (ch.isalpha() or ch.isspace()) for ch in t)
    if non_letters / max(len(t), 1) > 0.35:
        return True

    if re.search(r"(.)\1{5,}", t):  
        return True

    vowels = sum(ch in "aeiou" for ch in t)
    if vowels / max(letters, 1) < 0.18:
        return True

    tokens = re.findall(r"[a-z]+", t)
    if len(tokens) < 3:
        return True

    return False

@st.cache_data
def load_parts():
    return pd.read_csv(PARTS_PATH)

@st.cache_resource
def load_model():
    if MODEL_PATH.exists():
        return joblib.load(MODEL_PATH)
    return None


# Styling CSS

st.markdown(
    """
    <style>
      /* Background */
      .stApp {
        background: radial-gradient(1200px 800px at 20% 10%, rgba(0, 190, 255, 0.18), rgba(0,0,0,0)),
                    radial-gradient(1000px 700px at 80% 20%, rgba(0, 150, 255, 0.12), rgba(0,0,0,0)),
                    linear-gradient(180deg, #060a12 0%, #070b14 40%, #05070d 100%);
        color: #e7eefc;
      }

      section.main > div { padding-top: 1.1rem; }

      /* "Navbar" */
      .nav {
        display:flex;
        align-items:center;
        justify-content:space-between;
        padding: 0.6rem 0.2rem 1.1rem 0.2rem;
      }
      .brand {
        display:flex;
        align-items:center;
        gap:0.65rem;
        font-weight:800;
        letter-spacing:0.5px;
      }
      .brand .dot {
        width: 34px;
        height: 34px;
        border-radius: 10px;
        background: linear-gradient(135deg, rgba(0,190,255,1), rgba(0,120,255,1));
        display:flex;
        align-items:center;
        justify-content:center;
        box-shadow: 0 10px 30px rgba(0,160,255,0.25);
      }
      .brand .name { font-size: 1.1rem; }
      .navlinks {
        display:flex;
        gap:1.2rem;
        opacity: 0.86;
        font-weight: 600;
        font-size: 0.95rem;
      }
      .navcta { display:flex; gap:0.6rem; align-items:center; }
      .chip {
        padding: 0.45rem 0.75rem;
        border-radius: 999px;
        border: 1px solid rgba(255,255,255,0.10);
        background: rgba(255,255,255,0.04);
        font-size: 0.9rem;
      }
      .btn-primary {
        padding: 0.55rem 0.95rem;
        border-radius: 999px;
        background: linear-gradient(135deg, rgba(0,190,255,1), rgba(0,120,255,1));
        color: #041018;
        font-weight: 800;
        box-shadow: 0 10px 30px rgba(0,160,255,0.25);
      }

      /* Hero */
      .hero-title {
        font-size: 3.3rem;
        line-height: 1.05;
        margin: 0;
        font-weight: 900;
        letter-spacing: -0.8px;
      }
      .accent {
        background: linear-gradient(135deg, rgba(0,190,255,1), rgba(0,120,255,1));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
      }
      .hero-sub {
        margin-top: 1rem;
        font-size: 1.05rem;
        opacity: 0.86;
        max-width: 40rem;
      }

      /* Card */
      .card {
        border: 1px solid rgba(255,255,255,0.10);
        background: rgba(255,255,255,0.04);
        border-radius: 18px;
        padding: 1.25rem;
        box-shadow: 0 20px 50px rgba(0,0,0,0.35);
      }
      .card h3 { margin: 0 0 0.35rem 0; font-size: 1.35rem; }
      .muted { opacity: 0.75; }

      /* Make Streamlit widgets blend a bit */
      div.stButton > button {
        border-radius: 999px !important;
        padding: 0.65rem 1rem !important;
        border: 1px solid rgba(255,255,255,0.12) !important;
        background: rgba(255,255,255,0.06) !important;
        color: #e7eefc !important;
        font-weight: 800 !important;
      }
      div.stButton > button:hover {
        border-color: rgba(0,190,255,0.55) !important;
        background: rgba(0,190,255,0.10) !important;
      }

      /* Hide Streamlit header/footer */
      header, footer { visibility: hidden; }

      /* Clean up default markdown spacing slightly */
      .block-container { padding-left: 3rem; padding-right: 3rem; }

    </style>
    """,
    unsafe_allow_html=True,
)


# Navbar 

st.markdown(
    """
    <div class="nav">
      <div class="brand">
        <div class="dot">🚗</div>
        <div class="name">AUTOINTEL</div>
      </div>
      <div class="navlinks">
        <div>Chat</div>
        <div>My Cars</div>
        <div>Diagnostics</div>
        <div>Pricing</div>
        <div>Contact</div>
      </div>
      <div class="navcta">
        <div class="chip">🌙</div>
        <div class="btn-primary">Get Started</div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# Loading data/model

try:
    parts_df = load_parts()
except FileNotFoundError:
    st.error("Missing file: data/parts.csv — create it first, then rerun Streamlit.")
    st.stop()

model = load_model()
if model is None:
    st.warning("AI model not found yet. Run: python3 train_model.py to generate models/problem_classifier.joblib")


# Hero Section

left, right = st.columns([1.2, 1], gap="large")

with left:
    st.markdown(
        """
        <p class="hero-title">
          Car Trouble?<br/>
          Smart <span class="accent">AI Care</span>
        </p>
        <div class="hero-sub">
          Your AI mechanic for fast diagnostics and compatible part recommendations — saving you time and money.
          Describe symptoms in plain English, and AutoIntel predicts the likely system and suggested parts to check.
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns([0.48, 0.52], gap="small")
    with c1:
        start_diag = st.button("Start Free Diagnosis →", use_container_width=True)
    with c2:
        quick_check = st.button("Try Quick Check →", use_container_width=True)

with right:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("<h3>Quick Vehicle Check</h3>", unsafe_allow_html=True)
    st.markdown('<div class="muted">Pick a vehicle and enter a short symptom description.</div>', unsafe_allow_html=True)

    make = st.selectbox("Make", sorted(parts_df["make"].dropna().unique()), key="make_quick")

    model_name = st.selectbox(
        "Model",
        sorted(parts_df.loc[parts_df["make"] == make, "model"].dropna().unique()),
        key="model_quick"
    )

    year = st.selectbox(
        "Year",
        sorted(parts_df.loc[(parts_df["make"] == make) & (parts_df["model"] == model_name), "year"].dropna().unique()),
        key="year_quick"
    )

    problem_text = st.text_input(
        "Describe the issue",
        placeholder="e.g., brakes squeaking when stopping",
        key="problem_quick"
    )

    run_now = st.button("Check Vehicle Now", use_container_width=True, key="run_quick")

    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<br/>", unsafe_allow_html=True)


# Main Diagnosis Section 

if start_diag or quick_check or run_now:
    st.markdown("## Diagnosis & Recommendations")

    if model is None:
        st.error("AI model not found. Run: python3 train_model.py")
        st.stop()

    text = (problem_text or "").strip().lower()

    if not text:
        st.warning("Please enter a symptom description to continue.")
        st.stop()

    if looks_like_gibberish(text):
        st.error(
            "That description looks unclear. Please describe symptoms in plain English.\n\n"
            "Examples:\n"
            "- brakes squeaking when stopping\n"
            "- engine overheating in traffic\n"
            "- battery dying and headlights flickering"
        )
        st.stop()

    # Predict top 3 suggestions
    probs = model.predict_proba([text])[0]
    classes = model.classes_

    top_idx = np.argsort(probs)[::-1][:3]
    top3 = [(classes[i], float(probs[i])) for i in top_idx]
    predicted_category, confidence = top3[0]

    # Confidence gate
    if confidence < CONFIDENCE_THRESHOLD:
        st.warning(
            f"Low confidence ({confidence*100:.1f}%). Add more detail: when it happens, warning lights, noises, smells."
        )
        with st.expander("See top 3 predicted categories"):
            for cat, p in top3:
                st.write(f"- **{cat.upper()}** — {p*100:.1f}%")
        st.stop()

    st.success(f"Predicted Issue Category: **{predicted_category.upper()}**")
    st.write(f"Confidence: **{confidence*100:.1f}%**")

    with st.expander("Top 3 predictions"):
        for cat, p in top3:
            st.write(f"- **{cat.upper()}** — {p*100:.1f}%")

    st.markdown("### Recommended compatible parts")

    recommendations = parts_df[
        (parts_df["category"] == predicted_category) &
        (parts_df["make"] == make) &
        (parts_df["model"] == model_name) &
        (parts_df["year"] == year)
    ].copy()

    if recommendations.empty:
        st.info("No matching parts for this vehicle/category in the dataset yet.")
    else:
        min_cost = recommendations["avg_cost_gbp"].min()
        max_cost = recommendations["avg_cost_gbp"].max()
        st.write(f"Estimated parts cost range: **£{int(min_cost)} – £{int(max_cost)}**")

        st.dataframe(
            recommendations[["part_name", "avg_cost_gbp"]]
            .rename(columns={"avg_cost_gbp": "Average Cost (£)"}),
            use_container_width=True
        )

    st.markdown("---")
    st.caption(
        "Disclaimer: AutoIntel provides guidance based on user-reported symptoms and a curated dataset. "
        "It is not a substitute for professional inspection."
    )

st.caption("AutoIntel © AI-Based Car Repair & Parts Recommendation System")
