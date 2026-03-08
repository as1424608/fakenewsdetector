import pickle

# Load the model and vectorizer from ../models/
with open("../models/fake_news_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("../models/vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

print("📰 Fake News Detector")
print("Type a news article/text to check. Type 'exit' to quit.\n")

while True:
    text = input("Enter news text: ").strip()
    if text.lower() == "exit":
        print("👋 Exiting Fake News Detector. Goodbye!")
        break

    # Transform and predict
    X = vectorizer.transform([text])
    prediction = model.predict(X)[0]

    # Clearer output
    if prediction == 1:
        print("✅ Looks like REAL news.\n")
    else:
        print("⚠️ Looks like FAKE news.\n")


