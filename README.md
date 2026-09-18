# 📈 AI Financial Market & News Sentiment Predictor

An end-to-end Natural Language Processing (NLP) and Machine Learning pipeline designed to analyze financial news headlines, compute market sentiment polarity, and forecast stock price movements.

---

## 📌 Overview & Business Impact

Financial markets react instantly to news announcements, earnings reports, and macroeconomic indicators. Manual sentiment analysis is too slow to process high-frequency news feeds.

This project implements an automated AI pipeline that:
1. Preprocesses financial news text using **TF-IDF n-gram vectorization**.
2. Evaluates sentiment polarity and maps news context to price direction probabilities.
3. Classifies headlines as **Bullish** (Price Surge) or **Bearish** (Price Decline) with **91.60% Accuracy** and **0.9849 ROC-AUC**.
4. Provides an interactive **Streamlit Web Dashboard** for real-time sentiment inference.

---

## ⚙️ System Architecture & NLP Pipeline

```
[ Financial News Feed ] ➔ [ TF-IDF Text Vectorizer ] ➔ [ Ensemble Classifier ] ➔ [ Market Trend & Probability ]
```

1. **Text Preprocessing:** Tokenization, stop-word removal, and TF-IDF n-gram (1, 2) feature extraction.
2. **Dataset:** 2,500 financial headlines across top tech equities (AAPL, NVDA, TSLA, MSFT, AMZN) and cryptocurrencies (BTC, ETH).
3. **Model Benchmarking:** Comparison of Logistic Regression, Random Forest, and Gradient Boosting Classifiers.
4. **Interactive App:** Streamlit application providing real-time sentiment probability metrics.

---

## 🏆 Model Performance Benchmark

| Algorithm | Accuracy | ROC-AUC Score |
| :--- | :---: | :---: |
| **Gradient Boosting Classifier** | **91.60%** | **0.9849** |
| **Random Forest Classifier** | 91.20% | 0.9849 |
| **Logistic Regression** | 91.20% | 0.9840 |

---

## 🚀 Getting Started

### 1. Prerequisites & Installation

Clone the repository and install required packages:

```bash
git clone https://github.com/murshidkazi15/AI-Market-Sentiment-Predictor.git
cd AI-Market-Sentiment-Predictor
pip install -r requirements.txt
```

### 2. Train Models

To regenerate the dataset and train the ML models:

```bash
python generate_dataset.py
python model_pipeline.py
```

### 3. Launch Interactive Web App

Run the Streamlit dashboard locally:

```bash
streamlit run app.py
```

---

## 📁 Repository Structure

```
AI-Market-Sentiment-Predictor/
│
├── app.py                      # Interactive Streamlit Web App
├── model_pipeline.py           # NLP Feature Extraction & Model Training Script
├── generate_dataset.py         # Financial Dataset Generator
├── financial_news_sentiment.csv# Financial News Dataset (2,500 samples)
├── requirements.txt            # Python Dependencies
├── models/                     # Saved Model & Vectorizer Artifacts (.pkl)
└── README.md                   # Project Documentation
```

---

## 👤 Author

- **Murshid Kazi** — *Data Science & Information Management Student at NOVA IMS*
- GitHub: [@murshidkazi15](https://github.com/murshidkazi15)
