# app.py  — Improved Fake News Detector
# Fixes added:
#   1. Google Fact Check API integration
#   2. Wikipedia/Wikidata named entity verification
#   3. Contradiction detector (hard veto logic)
#   4. Stricter semantic thresholds (UNVERIFIED zone added)

import os
import re
import time
import pickle
import requests
import feedparser
import numpy as np
import wikipedia

from flask import Flask, render_template, request
from sentence_transformers import SentenceTransformer, util


NEWSAPI_KEY           = os.environ.get("NEWS_API_KEY", "")
GOOGLE_FACTCHECK_KEY  = os.environ.get("GOOGLE_FACTCHECK_KEY", "")  
NEWSAPI_TIMEOUT       = 8          # seconds


SEMANTIC_STRONG       = 0.70       # >= this → STRONGLY VERIFIED
SEMANTIC_LIKELY       = 0.55       # >= this → LIKELY REAL
SEMANTIC_UNVERIFIED   = 0.40       # >= this → UNVERIFIED  (was being called REAL before — now fixed!)
                                   # < 0.40  → UNVERIFIED / LIKELY FAKE

MAX_NEWSAPI_RESULTS   = 10

# ---- Load ML model & vectorizer ----
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "fake_news_model.pkl")
VECT_PATH  = os.path.join(os.path.dirname(__file__), "..", "models", "vectorizer.pkl")

if not os.path.exists(MODEL_PATH) or not os.path.exists(VECT_PATH):
    raise SystemExit("Model or vectorizer not found in ../models/. Train the model first.")

with open(MODEL_PATH, "rb") as f:
    ml_model = pickle.load(f)

with open(VECT_PATH, "rb") as f:
    vectorizer = pickle.load(f)

print("Loading sentence-transformers model...")
sbert = SentenceTransformer("all-MiniLM-L6-v2")
print("Sentence-transformers loaded.")



def google_factcheck(claim_text):
    """
    Query Google Fact Check Tools API.
    Returns dict with verdict, source, url — or None if nothing found.
    Get a free API key: console.cloud.google.com → Enable 'Fact Check Tools API'
    """
    if not GOOGLE_FACTCHECK_KEY:
        return None  

    url = "https://factchecktools.googleapis.com/v1alpha1/claims:search"
    params = {
        "query": claim_text[:200],  
        "languageCode": "en",
        "key": GOOGLE_FACTCHECK_KEY
    }
    try:
        r = requests.get(url, params=params, timeout=6)
        r.raise_for_status()
        data = r.json()
        claims = data.get("claims", [])
        if not claims:
            return None

        top = claims[0]
        reviews = top.get("claimReview", [])
        if not reviews:
            return None

        review     = reviews[0]
        rating     = review.get("textualRating", "Unknown")
        publisher  = (review.get("publisher") or {}).get("name", "Unknown source")
        review_url = review.get("url", "")

        return {
            "rating":    rating,
            "publisher": publisher,
            "url":       review_url,
            "raw":       top.get("text", "")
        }
    except Exception as e:
        return {"error": str(e)}




# Simple patterns to detect "X is [role] of [place]" claims
ROLE_PATTERNS = [
    r"(?P<name>[A-Z][a-z]+(?: [A-Z][a-z]+)*)\s+is\s+(?:the\s+)?(?P<role>prime minister|president|chief minister|governor|ceo|chairman|mayor|king|queen|chancellor)\s+of\s+(?P<place>[A-Za-z ]+)",
]

def extract_claim_entity(text):
    """Try to extract (name, role, place) from a claim using regex."""
    for pattern in ROLE_PATTERNS:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            return {
                "name":  m.group("name").strip(),
                "role":  m.group("role").strip().lower(),
                "place": m.group("place").strip().rstrip(".,!?")
            }
    return None


def wikipedia_verify(claim_text):
    """
    Extract a named entity claim, look it up on Wikipedia,
    check if the claimed person actually holds that role.
    Returns: status = 'CONTRADICTION' | 'SUPPORTED' | 'UNVERIFIED'
    """
    entity = extract_claim_entity(claim_text)
    if not entity:
        return {"status": "UNVERIFIED", "detail": "No verifiable entity claim detected"}

    query = "{role} of {place}".format(**entity)
    try:
        # wikipedia.summary raises DisambiguationError or PageError on failure
        summary = wikipedia.summary(query, sentences=3, auto_suggest=True)

        claimed_name = entity["name"].lower()

        # Check if the claimed name appears in the Wikipedia summary
        if claimed_name in summary.lower():
            return {
                "status": "SUPPORTED",
                "detail": "Wikipedia confirms '{name}' as {role} of {place}".format(**entity),
                "summary": summary[:300]
            }
        else:
            # Extract who Wikipedia says actually holds the role (first capitalized name)
            actual = re.findall(r"[A-Z][a-z]+(?: [A-Z][a-z]+)+", summary)
            actual_name = actual[0] if actual else "someone else"
            return {
                "status": "CONTRADICTION",
                "detail": "Wikipedia says the {role} of {place} is {actual}, NOT '{claimed}'".format(
                    role=entity["role"],
                    place=entity["place"],
                    actual=actual_name,
                    claimed=entity["name"]
                ),
                "summary": summary[:300]
            }
    except wikipedia.exceptions.DisambiguationError as e:
        # Try first option
        try:
            summary = wikipedia.summary(e.options[0], sentences=3)
            claimed_name = entity["name"].lower()
            status = "SUPPORTED" if claimed_name in summary.lower() else "CONTRADICTION"
            return {"status": status, "detail": summary[:300]}
        except Exception:
            return {"status": "UNVERIFIED", "detail": "Ambiguous Wikipedia result"}
    except Exception as e:
        return {"status": "UNVERIFIED", "detail": "Wikipedia lookup failed: " + str(e)}


def ml_predict(text):
    x    = vectorizer.transform([text])
    pred = ml_model.predict(x)[0]
    conf = None
    if hasattr(ml_model, "predict_proba"):
        try:
            probs = ml_model.predict_proba(x)[0]
            conf  = float(np.max(probs))
        except Exception:
            conf = None
    return int(pred), conf   # 1 → REAL, 0 → FAKE



def newsapi_search(query, page_size=MAX_NEWSAPI_RESULTS):
    if not NEWSAPI_KEY:
        return [], "Missing NEWS_API_KEY"
    url    = "https://newsapi.org/v2/everything"
    params = {"q": query, "language": "en", "pageSize": page_size,
              "sortBy": "relevancy", "apiKey": NEWSAPI_KEY}
    try:
        r = requests.get(url, params=params, timeout=NEWSAPI_TIMEOUT)
        r.raise_for_status()
        articles = r.json().get("articles", []) or []
        return [{"title": a.get("title") or "", "source": (a.get("source") or {}).get("name"),
                 "url": a.get("url"), "publishedAt": a.get("publishedAt")} for a in articles], None
    except Exception as e:
        return [], "NewsAPI error: " + str(e)


def google_news_rss_search(query, max_items=10):
    q   = requests.utils.quote(query)
    url = "https://news.google.com/rss/search?q={q}&hl=en-IN&gl=IN&ceid=IN:en".format(q=q)
    try:
        feed = feedparser.parse(url)
        return [{"title": e.get("title", ""), "source": (e.get("source") or {}).get("title"),
                 "url": e.get("link"), "publishedAt": e.get("published")}
                for e in feed.entries[:max_items]], None
    except Exception as e:
        return [], "Google RSS error: " + str(e)



def semantic_verify(user_text):
    articles, err = newsapi_search(user_text, page_size=MAX_NEWSAPI_RESULTS)
    if err:
        rss_articles, rss_err = google_news_rss_search(user_text)
        if rss_err:
            return {"status": "error", "message": err + " ; RSS: " + str(rss_err)}
        articles = rss_articles

    if not articles:
        return {"status": "no_results", "message": "No articles found"}

    titles = [a["title"] for a in articles if a.get("title")]
    if not titles:
        return {"status": "no_results", "message": "No titles returned"}

    try:
        user_emb   = sbert.encode(user_text, convert_to_tensor=True)
        titles_emb = sbert.encode(titles, convert_to_tensor=True)
        cos_scores = util.cos_sim(user_emb, titles_emb)[0].cpu().numpy()
    except Exception as e:
        return {"status": "error", "message": "Embedding error: " + str(e)}

    best_idx     = int(np.argmax(cos_scores))
    best_score   = float(cos_scores[best_idx])
    best_article = articles[best_idx]

    # FIX 4 — assign semantic tier
    if best_score >= SEMANTIC_STRONG:
        tier = "STRONGLY_VERIFIED"
    elif best_score >= SEMANTIC_LIKELY:
        tier = "LIKELY_REAL"
    elif best_score >= SEMANTIC_UNVERIFIED:
        tier = "UNVERIFIED"       
    else:
        tier = "UNVERIFIED_WEAK"

    return {
        "status":       "ok",
        "tier":         tier,
        "best_title":   best_article.get("title"),
        "best_source":  best_article.get("source"),
        "best_url":     best_article.get("url"),
        "score":        best_score
    }



def combine_decision(user_text):
    ml_label, ml_conf = ml_predict(user_text)
    verify            = semantic_verify(user_text)
    fact_check        = google_factcheck(user_text)       # FIX 1
    wiki_check        = wikipedia_verify(user_text)       # FIX 2

    # ---- Build verify message for UI ----
    verify_msg = ""
    if verify.get("status") == "ok":
        score = verify.get("score", 0.0)
        src   = verify.get("best_source") or "source"
        title = verify.get("best_title") or "N/A"
        tier  = verify.get("tier", "UNVERIFIED")
        if tier == "STRONGLY_VERIFIED":
            verify_msg = "Verified from {src} (similarity {s:.1f}%)".format(src=src, s=score*100)
        elif tier == "LIKELY_REAL":
            verify_msg = "Likely match: {src} (similarity {s:.1f}%)".format(src=src, s=score*100)
        else:
            verify_msg = "Weak match only: {t} (similarity {s:.1f}%) — insufficient to verify".format(t=title, s=score*100)
    elif verify.get("status") == "no_results":
        verify_msg = "No matching articles found"
    else:
        verify_msg = "Verification error: {m}".format(m=verify.get("message", ""))

    # ---- FIX 3: Hard veto — contradiction always wins ----
    #  Check Wikipedia contradiction first
    if wiki_check.get("status") == "CONTRADICTION":
        final = {
            "verdict":    "FAKE",
            "reason":     "CONTRADICTION: " + wiki_check.get("detail", ""),
            "confidence": 95.0,
            "veto":       True
        }
    # Check Google Fact Check hard-false ratings
    elif fact_check and "rating" in fact_check and \
         any(word in fact_check["rating"].lower() for word in ["false", "fake", "misleading", "pants on fire", "incorrect"]):
        final = {
            "verdict":    "FAKE",
            "reason":     "Fact-checked as '{r}' by {p}".format(
                              r=fact_check["rating"], p=fact_check["publisher"]),
            "confidence": 90.0,
            "veto":       True
        }
    else:
        # ---- Normal scoring flow ----
        sem_tier  = verify.get("tier", "UNVERIFIED") if verify.get("status") == "ok" else "UNVERIFIED"
        sem_score = verify.get("score", 0.0)         if verify.get("status") == "ok" else 0.0
        ml_conf_v = ml_conf if ml_conf is not None else 0.5

        # Map tier to numeric weight
        tier_weight = {
            "STRONGLY_VERIFIED": 1.0,
            "LIKELY_REAL":       0.7,
            "UNVERIFIED":        0.0,   # ← neutral, doesn't boost verdict
            "UNVERIFIED_WEAK":   0.0
        }.get(sem_tier, 0.0)

        # Boost from fact-check (if found and positive)
        fc_boost = 0.0
        if fact_check and "rating" in fact_check and \
           any(w in fact_check["rating"].lower() for w in ["true", "correct", "accurate", "mostly true"]):
            fc_boost = 0.10

        # Wikipedia support boost
        wiki_boost = 0.05 if wiki_check.get("status") == "SUPPORTED" else 0.0

        # Combined score (weighted)
        combined = (ml_conf_v * 0.45) + (tier_weight * sem_score * 0.45) + fc_boost + wiki_boost
        combined = min(combined, 1.0)

        if sem_tier in ("UNVERIFIED", "UNVERIFIED_WEAK") and ml_conf_v < 0.80:
            # Insufficient evidence — don't call it REAL
            verdict_str = "UNVERIFIED"
        elif ml_label == 1 and combined >= 0.65:
            verdict_str = "REAL (Verified)" if sem_tier == "STRONGLY_VERIFIED" else "LIKELY REAL"
        elif ml_label == 0 and ml_conf_v >= 0.70:
            verdict_str = "LIKELY FAKE (ML confident)"
        elif combined < 0.45:
            verdict_str = "LIKELY FAKE"
        else:
            verdict_str = "UNCERTAIN"

        final = {
            "verdict":    verdict_str,
            "reason":     "",
            "confidence": round(combined * 100, 1),
            "veto":       False
        }

    # ---- Compose full result for template ----
    return {
        "ml_label":      "REAL" if ml_label == 1 else "FAKE",
        "ml_confidence": round(float(ml_conf) * 100, 1) if ml_conf is not None else None,
        "verify_ok":     verify.get("tier") == "STRONGLY_VERIFIED",
        "verify_msg":    verify_msg,
        "verify_detail": verify,
        "wiki_check":    wiki_check,
        "fact_check":    fact_check,
        "final":         final
    }



app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    user_input = ""
    result     = None
    if request.method == "POST":
        user_input = (request.form.get("headline") or "").strip()
        if user_input:
            result = combine_decision(user_input)
    return render_template("index.html", user_input=user_input, result=result)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
