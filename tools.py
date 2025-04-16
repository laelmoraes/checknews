import pandas as pd
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk
import requests
from bs4 import BeautifulSoup
from datetime import datetime
from imblearn.over_sampling import SMOTE 
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('wordnet') 
STOPWORDS = set(stopwords.words('portuguese'))
LEMMATIZER = WordNetLemmatizer()
PUNCT_TABLE = str.maketrans("", "", string.punctuation)

def preprocess_text(text):
    text = text.lower().translate(PUNCT_TABLE) 
    words = text.split()  
    words = [LEMMATIZER.lemmatize(word) for word in words if word not in STOPWORDS]
    return " ".join(words)

def obter_noticia_link(url):
    try:
        resposta = requests.get(url)
        resposta.raise_for_status()
        soup = BeautifulSoup(resposta.content, 'html.parser')
        return soup.get_text()
    except requests.exceptions.RequestException as e:
        return None

def treinar_modelo():
    df = pd.read_excel("dt_fakenews.xlsx")
    df['clean_text'] = df['text'].apply(preprocess_text)
    
    vectorizer = TfidfVectorizer(max_features=5500, ngram_range=(1, 2))
    X = vectorizer.fit_transform(df['clean_text'])
    y = df['label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    model = MultinomialNB(alpha=0.5)
    model.fit(X_train_res, y_train_res)

    return model, vectorizer

def predict_fake_news(text, model, vectorizer):
    processed_text = preprocess_text(text)
    text_vectorized = vectorizer.transform([processed_text])
    prediction = model.predict(text_vectorized)
    return "FAKE NEWS" if prediction[0] == 1 else "REAL"
  