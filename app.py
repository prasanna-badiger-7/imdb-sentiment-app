import streamlit as st
import joblib
import re
from bs4 import BeautifulSoup

# Load model and vectorizer
model = joblib.load("sentiment_model_lr.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Cleaning function
def clean_text(text):
    text = BeautifulSoup(text, "html.parser").get_text()
    text = re.sub(r"[^a-zA-Z]", " ", text)
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text

# Prediction function
def predict_sentiment(review):
    cleaned = clean_text(review)
    vect = vectorizer.transform([cleaned])
    pred = model.predict(vect)[0]
    return "😊 Positive Review" if pred == 1 else "😞 Negative Review"

# Streamlit UI
st.title("🎬 IMDB Movie Review Sentiment Analyzer")

user_input = st.text_area("Paste your movie review here:")

if st.button("Analyze Sentiment"):
    if user_input.strip() == "":
        st.warning("Please enter some text!")
    else:
        result = predict_sentiment(user_input)
        st.subheader("Prediction:")
        st.success(result)
