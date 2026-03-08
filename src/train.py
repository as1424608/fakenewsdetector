import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle
import os
from tqdm import tqdm

# Download stopwords if not already
nltk.download("stopwords")

print("📂 Loading dataset...")
df = pd.read_csv("data/fake_or_real_news.csv")
print(f"✅ Dataset loaded with {len(df)} rows")

# Preprocessing
stop_words = set(stopwords.words("english"))

def clean_text(text):
    text = re.sub(r"[^a-zA-Z]", " ", text)  # keep only letters
    text = text.lower()
    text = " ".join([word for word in text.split() if word not in stop_words])
    return text

print("🧹 Cleaning text with progress bar...")
tqdm.pandas()   # enables progress_apply
df["text"] = df["text"].astype(str).progress_apply(clean_text)
print("✅ Text cleaned")

# Split dataset
print("✂️ Splitting dataset into train/test...")
X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["label"], test_size=0.2, random_state=42
)
print(f"✅ Training samples: {len(X_train)}, Testing samples: {len(X_test)}")

# TF-IDF
print("🔠 Converting text to TF-IDF vectors...")
vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)
print("✅ TF-IDF conversion done")

# Train model
print("🤖 Training Logistic Regression model...")
model = LogisticRegression(max_iter=200)
model.fit(X_train_tfidf, y_train)
print("✅ Training completed")

# Evaluate
y_pred = model.predict(X_test_tfidf)
accuracy = accuracy_score(y_test, y_pred)
print(f"🎯 Model trained with accuracy: {accuracy:.2f}")

# Save model + vectorizer
print("💾 Saving model and vectorizer...")
os.makedirs("../models", exist_ok=True)
with open("../models/fake_news_model.pkl", "wb") as f:
    pickle.dump(model, f)
with open("../models/vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)
print("✅ Model and vectorizer saved in ../models/")

