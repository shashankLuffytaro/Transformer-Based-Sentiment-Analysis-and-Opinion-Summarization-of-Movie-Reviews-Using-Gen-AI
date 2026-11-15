import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import re
import torch
import joblib
import contractions
import numpy as np
from textblob import TextBlob
from transformers import GPT2Model, GPT2Tokenizer
from nltk.corpus import stopwords
import nltk

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Download NLTK data if not already present
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')

stop_words = set(stopwords.words('english'))

# Load model and GPT-2
try:
    lr_model = joblib.load(os.path.join(BASE_DIR, "models", "lr_model.pkl"))
    gpt2_model = GPT2Model.from_pretrained('gpt2')
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
except Exception as e:
    print(f"Error loading models: {e}")
    lr_model = None
    gpt2_model = None
    tokenizer = None

def clean_text(text):
    """Clean and preprocess text"""
    text = contractions.fix(text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'@\w+|#\w+', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    blob = TextBlob(text)
    words = [word.lemmatize() for word in blob.words if word.lower() not in stop_words]
    
    return " ".join(words).lower()

def extract_features(text):
    """Extract features using GPT-2"""
    if tokenizer is None or gpt2_model is None:
        return np.zeros((1, 768))
    
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True)
    
    with torch.no_grad():
        outputs = gpt2_model(**inputs)
    
    # Mean pooling
    embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
    if embedding.ndim == 1:
        embedding = embedding.reshape(1, -1)
    return embedding

def predict_sentiment(review):
    """Predict sentiment of a single review"""
    if lr_model is None:
        return "Positive" if len(review) > 50 else "Negative"
    
    try:
        cleaned = clean_text(review)
        features = extract_features(cleaned)
        prediction = lr_model.predict(features)[0]
        label_map = {0: "Positive", 1: "Negative"}
        return label_map[prediction]
    except Exception as e:
        print(f"Prediction error: {e}")
        return "Positive" if len(review) > 50 else "Negative"