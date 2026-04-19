"""
NLP & Machine Learning Training Pipeline for Financial Market Sentiment
Author: Murshid Kazi (NOVA IMS)
"""

import os
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix

def train_sentiment_model(dataset_path="financial_news_sentiment.csv"):
    print(f"[*] Loading financial news dataset from {dataset_path}...")
    df = pd.read_csv(dataset_path)
    
    X_text = df["Headline"]
    y = df["Market_Movement"]
    
    # Stratified Train-Test Split (80% Train, 20% Test)
    X_train_text, X_test_text, y_train, y_test = train_test_split(
        X_text, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # TF-IDF Feature Extraction
    print("[*] Performing TF-IDF Text Vectorization...")
    vectorizer = TfidfVectorizer(max_features=1500, stop_words="english", ngram_range=(1, 2))
    X_train_vec = vectorizer.fit_transform(X_train_text)
    X_test_vec = vectorizer.transform(X_test_text)
    
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42)
    }
    
    print("\n" + "="*55)
    print("        FINANCIAL SENTIMENT MODEL EVALUATION RESULTS        ")
    print("="*55)
    
    best_model = None
    best_acc = 0.0
    best_name = ""
    
    for name, model in models.items():
        model.fit(X_train_vec, y_train)
        y_pred = model.predict(X_test_vec)
        y_prob = model.predict_proba(X_test_vec)[:, 1]
        
        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)
        
        print(f"\n▶ {name}:")
        print(f"  - Test Accuracy : {acc * 100:.2f}%")
        print(f"  - ROC-AUC Score : {auc:.4f}")
        
        if acc > best_acc:
            best_acc = acc
            best_model = model
            best_name = name
            
    print("\n" + "="*55)
    print(f"[*] Best Performing Model: {best_name} ({best_acc*100:.2f}% Accuracy)")
    
    # Save Model & Vectorizer
    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)
    
    with open(os.path.join(model_dir, "sentiment_model.pkl"), "wb") as f:
        pickle.dump(best_model, f)
        
    with open(os.path.join(model_dir, "tfidf_vectorizer.pkl"), "wb") as f:
        pickle.dump(vectorizer, f)
        
    print(f"[*] Saved model and vectorizer artifacts to '{model_dir}/'")
    return best_model, vectorizer

if __name__ == "__main__":
    train_sentiment_model()
