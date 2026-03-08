# app.py  (paste into ~/Desktop/fake-news-detector/src/app.py)
import os
import time
import pickle
import requests
import feedparser
import numpy as np

from flask import Flask, render_template, request
from sentence_transformers import SentenceTransformer, util

# ---- Config ----
NEWSAPI_KEY = os.environ.get("NEWS_API_KEY", "")
NEWSAPI_TIMEOUT = 8  # seconds
SEMANTIC_THRESHOLD = 0.70  # cosine similarity threshold (0..1)
MAX_NEWSAPI_RESULTS = 10

# ---- Load model & transformer ----
# model/vectorizer saved by your train.py into ../models/
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "fake_news_model.pkl")
VECT_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "vectorizer.pkl")

if not os.path.exists(MODEL_PATH) or not os.path.exists(VECT_PATH):
    raise SystemExit("Model or vectorizer not found in ../models/. Train the model first.")

with open(MODEL_PATH, "rb") as f:
    ml_model = pickle.load(f)

with open(VECT_PATH, "rb") as f:
    vectorizer = pickle.load(f)

# sentence-transformers model (semantic embeddings)
print("Loading sentence-transformers model (this may download the model on first run)...")
sbert = SentenceTransformer("all-MiniLM-L6-v2")
print("Sentence-transformers loaded.")

# ---- Helpers: ML prediction ----
def ml_predict(text):
    x = vectorizer.transform([text])
    pred = ml_model.predict(x)[0]
    conf = None
    if hasattr(ml_model, "predict_proba"):
        try:
            probs = ml_model.predict_proba(x)[0]
            conf = float(np.max(probs))
        except Exception:
            conf = None
    return int(pred), conf  # label: 1 => REAL, 0 => FAKE

# ---- Helpers: NewsAPI (archive + live) ----
def newsapi_search(query, page_size=MAX_NEWSAPI_RESULTS):
    """Search NewsAPI /everything for the exact query (archive + recent)"""
    if not NEWSAPI_KEY:
        return [], "Missing NEWS_API_KEY (set with export NEWS_API_KEY=...)"

    url = "https://newsapi.org/v2/everything"
    params = {
        "q": query,
        "language": "en",
        "pageSize": page_size,
        "sortBy": "relevancy",
        "apiKey": NEWSAPI_KEY
    }
    try:
        r = requests.get(url, params=params, timeout=NEWSAPI_TIMEOUT)
        r.raise_for_status()
        data = r.json()
        articles = data.get("articles", []) or []
        out = []
        for a in articles:
            out.append({
                "title": a.get("title") or "",
                "source": (a.get("source") or {}).get("name"),
                "url": a.get("url"),
                "publishedAt": a.get("publishedAt")
            })
        return out, None
    except Exception as e:
        return [], "NewsAPI error: " + str(e)

# ---- Helpers: Google News RSS fallback ----
def google_news_rss_search(query, max_items=10):
    q = requests.utils.quote(query)
    url = "https://news.google.com/rss/search?q={q}&hl=en-IN&gl=IN&ceid=IN:en".format(q=q)
    try:
        feed = feedparser.parse(url)
        out = []
        for e in feed.entries[:max_items]:
            out.append({
                "title": e.get("title", ""),
                "source": (e.get("source") or {}).get("title"),
                "url": e.get("link"),
                "publishedAt": e.get("published")
            })
        return out, None
    except Exception as e:
        return [], "Google RSS error: " + str(e)

# ---- Semantic verification ----
def semantic_verify(user_text):
    """
    Search news APIs (NewsAPI everything + Google RSS),
    compute semantic similarity (SBERT) and return best match info.
    """
    # 1) prefer NewsAPI archive search first (better content)
    articles, err = newsapi_search(user_text, page_size=MAX_NEWSAPI_RESULTS)
    if err:
        # try RSS fallback
        rss_articles, rss_err = google_news_rss_search(user_text, max_items=MAX_NEWSAPI_RESULTS)
        if rss_err:
            # return error so UI can show fallback / warn
            return {"status": "error", "message": err + " ; RSS: " + str(rss_err)}
        articles = rss_articles  # use rss results

    if not articles:
        return {"status": "no_results", "message": "No articles found"}

    # Extract titles for embedding
    titles = [a["title"] for a in articles if a.get("title")]
    if not titles:
        return {"status": "no_results", "message": "No titles returned"}

    # Encode once
    try:
        user_emb = sbert.encode(user_text, convert_to_tensor=True)
        titles_emb = sbert.encode(titles, convert_to_tensor=True)
        cos_scores = util.cos_sim(user_emb, titles_emb)[0].cpu().numpy()  # numpy array
    except Exception as e:
        return {"status": "error", "message": "Embedding error: " + str(e)}

    # best match
    best_idx = int(np.argmax(cos_scores))
    best_score = float(cos_scores[best_idx])
    best_article = articles[best_idx]

    result = {
        "status": "ok",
        "best_title": best_article.get("title"),
        "best_source": best_article.get("source"),
        "best_url": best_article.get("url"),
        "score": best_score
    }
    return result

# ---- Decision combining ----
def combine_decision(user_text):
    ml_label, ml_conf = ml_predict(user_text)  # 1 real, 0 fake
    verify = semantic_verify(user_text)

    # interpret verify
    verified = False
    verify_msg = ""
    if verify.get("status") == "ok":
        score = verify.get("score", 0.0)
        if score >= SEMANTIC_THRESHOLD:
            verified = True
            verify_msg = "Verified from {src} (similarity {s:.1f}%)".format(src=verify.get("best_source") or "source", s=score * 100)
        else:
            verify_msg = "Best match: {t} (similarity {s:.1f}%)".format(t=verify.get("best_title") or "N/A", s=score * 100)
    elif verify.get("status") == "no_results":
        verify_msg = "No matching articles found"
    else:
        verify_msg = "Verification error: {m}".format(m=verify.get("message"))

    # Combine rules for final verdict (simple logic, tuneable)
    if verified:
        final = {"verdict": "REAL (Verified)", "confidence": max(ml_conf or 0.0, verify.get("score", 0.0))}
    else:
        # If ml says real with high confidence, consider likely real
        if ml_label == 1 and (ml_conf is not None and ml_conf >= 0.70):
            final = {"verdict": "LIKELY REAL (ML confident)", "confidence": float(ml_conf)}
        elif ml_label == 0 and (ml_conf is not None and ml_conf >= 0.70):
            final = {"verdict": "LIKELY FAKE (ML confident)", "confidence": float(ml_conf)}
        else:
            final = {"verdict": "UNCERTAIN", "confidence": float(ml_conf or 0.0)}

    # Compose structured result
    return {
        "ml_label": "REAL" if ml_label == 1 else "FAKE",
        "ml_confidence": float(ml_conf) if ml_conf is not None else None,
        "verify_ok": verified,
        "verify_msg": verify_msg,
        "verify_detail": verify,
        "final": final
    }

# ---- Flask app ----
app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    user_input = ""
    result = None
    if request.method == "POST":
        user_input = (request.form.get("headline") or "").strip()
        if user_input:
            result = combine_decision(user_input)
    return render_template("index.html", user_input=user_input, result=result)

if __name__ == "__main__":
    # bind 0.0.0.0 to allow other devices on LAN (optional)
    app.run(host="0.0.0.0", port=5000, debug=True)

