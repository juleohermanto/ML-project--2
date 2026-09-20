import os
import joblib
import re
from transformers import pipeline

# Base path for models
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

# Global variables to hold models in memory
tfidf = None
aspect_mlb = None
aspect_model = None
indobert_pipeline = None
slang_dict = None

# IndoBERT label mapping (LabelEncoder used alphabetical order in Colab)
LABEL_MAP = {
    "LABEL_0": "Negative",
    "LABEL_1": "Neutral",
    "LABEL_2": "Positive"
}

def load_models():
    """Loads the trained artifacts and IndoBERT from disk into memory."""
    global tfidf, aspect_mlb, aspect_model, indobert_pipeline, slang_dict
    
    print("Loading ML models into memory...")
    try:
        # Load Baseline Aspect Models (since we only trained Sentiment on IndoBERT)
        tfidf = joblib.load(os.path.join(MODEL_DIR, "tfidf_vectorizer.joblib"))
        aspect_mlb = joblib.load(os.path.join(MODEL_DIR, "aspect_mlb.joblib"))
        aspect_model = joblib.load(os.path.join(MODEL_DIR, "aspect_model.joblib"))
        
        # Load Deep Learning Sentiment Model (IndoBERT)
        indobert_path = os.path.join(MODEL_DIR, "indobert_model")
        print(f"Loading IndoBERT from {indobert_path}...")
        indobert_pipeline = pipeline("text-classification", model=indobert_path, tokenizer=indobert_path)
        
        print("All models loaded successfully!")
    except Exception as e:
        print(f"WARNING: Model loading failed. Error: {e}")
        
    # Replicate the slang dictionary from Phase 2
    slang_dict = {
        "bgt": "banget", "bgs": "bagus", "tdk": "tidak", "gpp": "tidak apa-apa",
        "spt": "seperti", "pake": "pakai", "ga": "tidak", "gak": "tidak",
        "ori": "original", "packing": "pengemasan", "pengirimannya": "pengiriman",
        "apa2": "apa-apa"
    }

def clean_and_normalize(text: str) -> str:
    text = str(text).lower()  
    text = re.sub(r'<.*?>', ' ', text)  
    text = re.sub(r'http\S+|www\S+|https\S+', ' ', text, flags=re.MULTILINE)  
    text = re.sub(r'[^a-z0-9\s]', ' ', text)  
    text = re.sub(r'\s+', ' ', text).strip()  
    
    words = text.split()
    normalized_words = [slang_dict.get(word, word) for word in words]
    return " ".join(normalized_words)

def predict_pipeline(raw_text: str):
    """Runs inference using IndoBERT for sentiment, and TF-IDF for aspects."""
    if indobert_pipeline is None:
        raise RuntimeError("Models are not loaded!")

    # 1. Preprocess
    clean_text = clean_and_normalize(raw_text)
    
    # 2. Predict Sentiment using Deep Learning (IndoBERT)
    # Pipeline returns a list of dicts, e.g., [{'label': 'LABEL_0', 'score': 0.99}]
    bert_result = indobert_pipeline(clean_text)[0]
    raw_label = bert_result['label']
    sentiment_pred = LABEL_MAP.get(raw_label, raw_label)
    confidence = float(bert_result['score'])
    
    # 3. Predict Aspects using traditional ML
    vec = tfidf.transform([clean_text])
    aspect_bin = aspect_model.predict(vec)
    aspects_pred = aspect_mlb.inverse_transform(aspect_bin)[0]
    
    return {
        "sentiment": sentiment_pred,
        "confidence": float(round(confidence, 2)),
        "aspects": list(aspects_pred)
    }
