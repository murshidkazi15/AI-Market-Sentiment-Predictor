"""
Synthetic Financial News Sentiment & Market Movement Dataset Generator
Generates realistic financial news text data with sentiment polarity and target price direction.
"""

import pandas as pd
import numpy as np

def generate_financial_dataset(output_path="financial_news_sentiment.csv", num_samples=2500):
    np.random.seed(42)
    
    tickers = ["AAPL", "TSLA", "NVDA", "AMZN", "MSFT", "GOOGL", "META", "BTC", "ETH"]
    
    bullish_templates = [
        "{ticker} reports record quarterly revenue exceeding Wall Street expectations.",
        "{ticker} launches groundbreaking AI product line, driving analyst upgrades.",
        "Analysts raise price target for {ticker} following strong demand metrics.",
        "{ticker} secures major enterprise partnership, expanding market dominance.",
        "Institutional investors surge buying in {ticker} after bullish earnings call.",
        "Strong subscriber growth drives {ticker} shares to new 52-week high.",
        "{ticker} announces strategic acquisition to boost AI infrastructure.",
        "Federal Reserve signals rate cut interest, sparking market rally in {ticker}."
    ]
    
    bearish_templates = [
        "{ticker} misses Q3 earnings estimates amid rising operational costs.",
        "Regulatory scrutiny intensifies for {ticker}, causing stock slump.",
        "{ticker} faces supply chain bottlenecks, lowering annual guidance.",
        "Analyst downgrade for {ticker} citing margin compression and slow growth.",
        "{ticker} shares plummet following unexpected CEO departure and audit concerns.",
        "Decreased consumer demand hits {ticker} quarterly profit margins.",
        "{ticker} reports cybersecurity breach, prompting investor selloff.",
        "Inflation fears and macro headwinds pressure {ticker} valuations lower."
    ]
    
    neutral_templates = [
        "{ticker} schedules upcoming annual shareholder meeting for next month.",
        "{ticker} holds steady as investors await upcoming inflation report.",
        "Market analysts maintain neutral rating on {ticker} ahead of earnings.",
        "{ticker} announces routine board member appointment.",
        "{ticker} trades sideways amidst broader market consolidation."
    ]
    
    data = []
    for i in range(num_samples):
        ticker = np.random.choice(tickers)
        label_type = np.random.choice(["bullish", "bearish", "neutral"], p=[0.45, 0.40, 0.15])
        
        if label_type == "bullish":
            text = np.random.choice(bullish_templates).format(ticker=ticker)
            sentiment_score = np.random.uniform(0.35, 0.95)
            market_movement = 1 # Price Up
        elif label_type == "bearish":
            text = np.random.choice(bearish_templates).format(ticker=ticker)
            sentiment_score = np.random.uniform(-0.95, -0.35)
            market_movement = 0 # Price Down
        else:
            text = np.random.choice(neutral_templates).format(ticker=ticker)
            sentiment_score = np.random.uniform(-0.15, 0.15)
            market_movement = 1 if np.random.rand() > 0.5 else 0
            
        # Add random noise/variations
        daily_return_pct = round(sentiment_score * 3.5 + np.random.normal(0, 0.8), 2)
        
        data.append({
            "Headline_ID": i + 1001,
            "Ticker": ticker,
            "Headline": text,
            "Sentiment_Score": round(sentiment_score, 4),
            "Daily_Return_Pct": daily_return_pct,
            "Market_Movement": market_movement
        })
        
    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)
    print(f"[*] Generated dataset with {num_samples} samples saved to: {output_path}")
    return df

if __name__ == "__main__":
    generate_financial_dataset()
