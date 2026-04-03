import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
from models.model_loader import load_models
from utils.analyzer import analyze_text
from utils.visualization import plot_confidence

st.set_page_config(page_title="Multi-Task NLP Analyzer", page_icon="🤖")

st.title("🤖 Multi-Task NLP Analyzer")
st.write("Sentiment + Emotion + Toxicity Detection using Hugging Face Transformers")

# Load models (cached)
@st.cache_resource
def get_models():
    return load_models()

sentiment_model, emotion_model, toxicity_model = get_models()

# Input box
user_text = st.text_area("✍️ Enter your text:")

if st.button("Analyze"):
    if user_text.strip():
        result = analyze_text(user_text, sentiment_model, emotion_model, toxicity_model)

        st.subheader("📊 Results")

        col1, col2, col3 = st.columns(3)

        col1.metric("😊 Sentiment", result["sentiment"], f"{result['sentiment_score']:.2f}")
        col2.metric("😡 Emotion", result["emotion"], f"{result['emotion_score']:.2f}")
        col3.metric("☣️ Toxicity", result["toxicity"], f"{result['toxicity_score']:.2f}")

        # Show graph
        fig = plot_confidence(result)
        st.pyplot(fig)

    else:
        st.warning("Please enter some text!")