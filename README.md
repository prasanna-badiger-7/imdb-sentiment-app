# 🎬 IMDB Movie Review Sentiment Analysis

Classify IMDB movie reviews as **positive** or **negative** using machine learning algorithms built with **Scikit-learn**. The best-performing model, **Logistic Regression**, is deployed using a Streamlit-based web interface.

---

## 🚀 Project Overview

- 📚 **Dataset:** [IMDB Large Movie Review Dataset](https://ai.stanford.edu/~amaas/data/sentiment/)
- 🔤 **Task:** Binary sentiment classification (positive or negative)
- ⚙️ **Tech Stack:** Python, Scikit-learn, TF-IDF, Streamlit, Joblib
- 🧠 **Models Compared:** Logistic Regression, Linear SVC, Naive Bayes, Random Forest, Decision Tree

---

## 🧾 Dataset Details

- 50,000 movie reviews in total  
  - 25,000 training samples  
  - 25,000 testing samples  
- Binary labels:  
  - `1` → Positive  
  - `0` → Negative

---

## 🛠️ Project Pipeline

### 1. Data Cleaning
- Removed HTML tags using `BeautifulSoup`
- Removed non-alphabet characters
- Lowercased text and normalized whitespace

### 2. Feature Extraction
- Used `TfidfVectorizer` with:
  - Max features: 10,000
  - N-grams: unigrams and bigrams
  - English stopword removal

### 3. Model Training
Trained and evaluated 5 classifiers:

| Model               | Accuracy |
|---------------------|----------|
| ✅ **Logistic Regression** | **88.70%** |
| Linear SVC          | 88.16%   |
| Naive Bayes         | 86.52%   |
| Random Forest       | 83.98%   |
| Decision Tree       | 71.34%   |

### 4. Model Persistence
- Saved the trained model and TF-IDF vectorizer using `joblib`

### 5. Prediction Function
- Function to predict sentiment of any custom movie review

---

## 💻 Streamlit App

### 🔮 Features
- Paste or type a review
- Get instant sentiment classification with emoji
- Clean and simple UI

> 🔗 [Live App URL](https://prasanna-badiger-7-imdb-sentiment-app.streamlit.app/)

---

## 🧠 Key Learnings

- Practical understanding of text vectorization using TF-IDF
- Model comparison for NLP sentiment classification
- Preprocessing pipeline using real-world data
- Streamlit app development and user interactivity

---

## 📂 Project Structure

```
imdb-sentiment-app/
├── app.py                    # Streamlit frontend
├── sentiment_model_lr.pkl    # Trained Logistic Regression model
├── tfidf_vectorizer.pkl      # Saved TF-IDF vectorizer
├── requirements.txt          # Python dependencies
├── README.md                 # Project summary
├── sentiment_analysis.ipynb  # Jupyter notebook of training models
```

---

## ▶️ How to Run Locally

```bash
git clone https://github.com/prasanna-badiger-7/imdb-sentiment-app.git
cd imdb-sentiment-app

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

---

## 📌 Requirements

```
streamlit
scikit-learn
joblib
beautifulsoup4
```

---

## 📧 Contact

Prasanna
[GitHub](https://github.com/prasanna-badiger-7)