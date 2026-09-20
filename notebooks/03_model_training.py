# %% [markdown]
# # Phase 3: Model Building & Evaluation
# This script trains the baseline Machine Learning models for Sentiment and Aspect classification.
# It uses TF-IDF for text vectorization and Logistic Regression as the baseline classifier.

# %%
import os
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import classification_report, confusion_matrix

# Define file paths
TRAIN_DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "train_data.csv")
TEST_DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "test_data.csv")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

# %%
# 1. Load the Split Data
print("Loading train and test data...")
train_df = pd.read_csv(TRAIN_DATA_PATH)
test_df = pd.read_csv(TEST_DATA_PATH)

# Handle empty text just in case
train_df['normalized_text'] = train_df['normalized_text'].fillna('')
test_df['normalized_text'] = test_df['normalized_text'].fillna('')

X_train = train_df['normalized_text']
X_test = test_df['normalized_text']

# Target for Sentiment
y_sentiment_train = train_df['sentiment']
y_sentiment_test = test_df['sentiment']

# %%
# 2. Text Vectorization (TF-IDF Baseline)
# We use unigrams and bigrams to capture context like "tidak bagus" (not good)
print("\nVectorizing text data (TF-IDF)...")
tfidf = TfidfVectorizer(ngram_range=(1, 2), max_features=5000)

X_train_vec = tfidf.fit_transform(X_train)
X_test_vec = tfidf.transform(X_test)

print(f"TF-IDF feature matrix shape (Train): {X_train_vec.shape}")

# %%
# 3. Train Sentiment Classifier (Multi-class)
print("\nTraining Sentiment Classifier (Logistic Regression)...")
sentiment_model = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
sentiment_model.fit(X_train_vec, y_sentiment_train)

# Evaluate Sentiment Model
y_sentiment_pred = sentiment_model.predict(X_test_vec)
print("\n--- Sentiment Model Evaluation ---")
print(classification_report(y_sentiment_test, y_sentiment_pred))

# %%
# 4. Train Aspect Classifier (Multi-Label)
print("\nPreparing Aspect Multi-Label Data...")
# Aspects are stored as comma-separated strings (e.g., "Shipping/Delivery,Product Quality")
def parse_aspects(aspect_str):
    if pd.isna(aspect_str): return []
    return [a.strip() for a in str(aspect_str).split(",")]

y_aspect_train = train_df['aspects'].apply(parse_aspects)
y_aspect_test = test_df['aspects'].apply(parse_aspects)

# Convert lists of strings into binary matrices (1 if aspect is present, 0 if not)
mlb = MultiLabelBinarizer()
y_aspect_train_bin = mlb.fit_transform(y_aspect_train)
y_aspect_test_bin = mlb.transform(y_aspect_test)

print("Training Aspect Classifier (OneVsRest + Logistic Regression)...")
# We use OneVsRestClassifier because a single review can have MULTIPLE tags simultaneously
aspect_model = OneVsRestClassifier(LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42))
aspect_model.fit(X_train_vec, y_aspect_train_bin)

# Evaluate Aspect Model
y_aspect_pred = aspect_model.predict(X_test_vec)
print("\n--- Aspect Model Evaluation ---")
print(classification_report(y_aspect_test_bin, y_aspect_pred, target_names=mlb.classes_))

# %%
# 5. Model Serialization (Saving for Phase 4: FastAPI Backend)
print("\nSaving model artifacts to 'models/' directory...")
os.makedirs(MODEL_DIR, exist_ok=True)

joblib.dump(tfidf, os.path.join(MODEL_DIR, "tfidf_vectorizer.joblib"))
joblib.dump(sentiment_model, os.path.join(MODEL_DIR, "sentiment_model.joblib"))
joblib.dump(mlb, os.path.join(MODEL_DIR, "aspect_mlb.joblib"))
joblib.dump(aspect_model, os.path.join(MODEL_DIR, "aspect_model.joblib"))

print("Phase 3 Complete! Models exported successfully.")
