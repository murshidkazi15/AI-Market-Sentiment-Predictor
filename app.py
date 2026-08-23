"""
Interactive Financial Market Sentiment & Trend Prediction Web Application
Author: Murshid Kazi (NOVA IMS)
"""

import os
import pickle
import pandas as pd
import numpy as np
import streamlit as st

st.set_page_config(
    page_title="AI Financial Market Sentiment Predictor",
    page_icon="📈",
    layout="wide"
)

@st.cache_resource
def load_artifacts():
    model_path = os.path.join("models", "sentiment_model.pkl")
    vec_path = os.path.join("models", "tfidf_vectorizer.pkl")
    
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(vec_path, "rb") as f:
        vectorizer = pickle.load(f)
        
    return model, vectorizer

st.title("📈 AI Financial Market & News Sentiment Predictor")
st.markdown("""
*An end-to-end NLP & Machine Learning platform developed by **Murshid Kazi (NOVA IMS)** for analyzing financial headlines, predicting market sentiment polarity, and forecasting stock price directions.*
""")

try:
    model, vectorizer = load_artifacts()
    st.success("✅ Machine Learning Models & TF-IDF Vectorizer Loaded Successfully!")
except Exception as e:
    st.error(f"Error loading models: {e}")
    st.stop()

col1, col2 = st.columns([3, 2])

with col1:
    st.subheader("📰 Analyze Headline Sentiment")
    
    sample_headlines = [
        "Select a sample headline...",
        "NVDA launches groundbreaking AI product line, driving analyst upgrades.",
        "AAPL misses Q3 earnings estimates amid rising operational costs.",
        "Institutional investors surge buying in TSLA after bullish earnings call.",
        "Regulatory scrutiny intensifies for META, causing stock slump.",
        "BTC surges past resistance as global institutional demand reaches record high."
    ]
    
    selected_sample = st.selectbox("Choose a sample news headline or type your own below:", sample_headlines)
    
    default_text = "" if selected_sample == "Select a sample headline..." else selected_sample
    user_input = st.text_area("Financial News Headline / Article Snippet:", value=default_text, height=120)
    
    if st.button("🚀 Predict Market Direction", type="primary"):
        if not user_input.strip():
            st.warning("Please enter a valid news headline to analyze.")
        else:
            vec_input = vectorizer.transform([user_input])
            pred = model.predict(vec_input)[0]
            probs = model.predict_proba(vec_input)[0]
            
            bullish_prob = probs[1] * 100
            bearish_prob = probs[0] * 100
            
            st.markdown("---")
            st.subheader("🎯 Prediction & Sentiment Analysis")
            
            if pred == 1:
                st.markdown(f"### 🟢 Sentiment: **BULLISH** (Price Up Probability: **{bullish_prob:.1f}%**)")
                st.progress(probs[1])
            else:
                st.markdown(f"### 🔴 Sentiment: **BEARISH** (Price Down Probability: **{bearish_prob:.1f}%**)")
                st.progress(probs[0])
                
            m_col1, m_col2 = st.columns(2)
            m_col1.metric("Bullish Probability", f"{bullish_prob:.1f}%")
            m_col2.metric("Bearish Probability", f"{bearish_prob:.1f}%")

with col2:
    st.subheader("📊 Model Performance Summary")
    st.markdown("""
    | Classifier | Accuracy | ROC-AUC |
    | :--- | :---: | :---: |
    | **Gradient Boosting** | **91.60%** | **0.9849** |
    | **Random Forest** | **91.20%** | **0.9849** |
    | **Logistic Regression** | **91.20%** | **0.9840** |
    """)
    
    st.subheader("💡 Key Features")
    st.markdown("""
    - **NLP Vectorization:** TF-IDF n-gram (1,2) text feature extraction.
    - **Supervised Learning:** Gradient Boosting & Random Forest ensemble classifiers.
    - **Dataset Size:** 2,500 financial headlines across tech, crypto, and market indices.
    """)

st.markdown("---")
st.markdown("*Designed with ❤️ by Murshid Kazi — Data Science & Information Management @ NOVA IMS*")
