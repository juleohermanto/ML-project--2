# %% [markdown]
# # Phase 2: Indonesian Text Preprocessing Pipeline
# Rewritten to use standard Python libraries (csv, re, random) to avoid dependency issues.

# %%
import os
import re
import csv
import random

# Define file paths
RAW_DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "reviews_raw.csv")
CLEANED_DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "reviews_cleaned.csv")
TRAIN_DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "train_data.csv")
TEST_DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "test_data.csv")

# %%
# 1. Text Cleaning
def clean_text(text):
    text = str(text).lower()  
    text = re.sub(r'<.*?>', ' ', text)  
    text = re.sub(r'http\S+|www\S+|https\S+', ' ', text, flags=re.MULTILINE)  
    text = re.sub(r'[^a-z0-9\s]', ' ', text)  
    text = re.sub(r'\s+', ' ', text).strip()  
    return text

# %%
# 2. Slang Normalization
slang_dict = {
    "bgt": "banget", "bgs": "bagus", "tdk": "tidak", "gpp": "tidak apa-apa",
    "spt": "seperti", "pake": "pakai", "ga": "tidak", "gak": "tidak",
    "ori": "original", "packing": "pengemasan", "pengirimannya": "pengiriman",
    "apa2": "apa-apa"
}

def normalize_slang(text):
    words = text.split()
    normalized_words = [slang_dict.get(word, word) for word in words]
    return " ".join(normalized_words)

# %%
# 3. Labeling Strategies
def get_sentiment(rating):
    rating = int(rating)
    if rating <= 2: return "Negative"
    elif rating == 3: return "Neutral"
    else: return "Positive"

aspect_keywords = {
    "Shipping/Delivery": ["lama", "lambat", "kurir", "pengiriman", "cepat"],
    "Packaging/Condition": ["rusak", "penyok", "bocor", "bubble wrap", "aman", "hancur", "pengemasan"],
    "Product Quality": ["original", "tiruan", "bagus", "cacat", "ori", "utuh"],
    "Customer Service": ["admin", "respon", "ramah", "acuh"]
}

def get_aspects(text):
    aspects = []
    for aspect, keywords in aspect_keywords.items():
        if any(keyword in text for keyword in keywords):
            aspects.append(aspect)
    if not aspects:
        aspects.append("General")
    return ",".join(aspects)

# %%
# 4. Processing Pipeline
if __name__ == "__main__":
    print("Loading raw data...")
    processed_rows = []
    
    with open(RAW_DATA_PATH, mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            clean_val = clean_text(row['review_text'])
            norm_val = normalize_slang(clean_val)
            sentiment = get_sentiment(row['star_rating'])
            aspects = get_aspects(norm_val)
            
            processed_rows.append({
                "review_text": row['review_text'],
                "star_rating": row['star_rating'],
                "timestamp": row['timestamp'],
                "clean_text": clean_val,
                "normalized_text": norm_val,
                "sentiment": sentiment,
                "aspects": aspects
            })
            
    print(f"Processed {len(processed_rows)} records.")
    
    # Save the cleaned full dataset
    fieldnames = ["review_text", "star_rating", "timestamp", "clean_text", "normalized_text", "sentiment", "aspects"]
    with open(CLEANED_DATA_PATH, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(processed_rows)
        
    print(f"Saved cleaned dataset to: {CLEANED_DATA_PATH}")

    # %%
    # 5. Stratified Train/Test Split (80/20)
    # Group by sentiment to ensure proportional splitting
    grouped_by_sentiment = {"Positive": [], "Neutral": [], "Negative": []}
    for row in processed_rows:
        grouped_by_sentiment[row['sentiment']].append(row)
        
    train_data, test_data = [], []
    
    random.seed(42) # Ensure reproducible split
    for sentiment, rows in grouped_by_sentiment.items():
        random.shuffle(rows)
        split_idx = int(len(rows) * 0.8)
        train_data.extend(rows[:split_idx])
        test_data.extend(rows[split_idx:])
        
    # Shuffle the final train and test sets
    random.shuffle(train_data)
    random.shuffle(test_data)
    
    with open(TRAIN_DATA_PATH, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(train_data)
        
    with open(TEST_DATA_PATH, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(test_data)
        
    print("\nData Splitting Complete!")
    print(f"Train set: {len(train_data)} samples")
    print(f"Test set: {len(test_data)} samples")
