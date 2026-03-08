import requests
import pandas as pd
from datetime import datetime, timedelta

# You need a NewsAPI key: https://newsapi.org/
API_KEY = "94dfc9e75d8447f3aa657aa0589b0d5f"
BASE_URL = "https://newsapi.org/v2/everything"

def fetch_old_news(query="India", days=30, page_size=100):
    """Fetch old news articles from NewsAPI"""
    all_articles = []
    today = datetime.now()
    from_date = (today - timedelta(days=days)).strftime("%Y-%m-%d")

    url = f"{BASE_URL}?q={query}&from={from_date}&sortBy=publishedAt&pageSize={page_size}&apiKey={API_KEY}"
    response = requests.get(url)
    data = response.json()

    if "articles" in data:
        for article in data["articles"]:
            all_articles.append({
                "title": article["title"],
                "description": article["description"],
                "publishedAt": article["publishedAt"],
                "source": article["source"]["name"]
            })
    return all_articles

if __name__ == "__main__":
    print("📥 Fetching old news (last 30 days)...")
    articles = fetch_old_news(query="India", days=30)

    # Save into CSV for later training/verification
    df = pd.DataFrame(articles)
    df.to_csv("../data/old_news.csv", index=False)
    print(f"✅ Saved {len(df)} old news articles into ../data/old_news.csv")
