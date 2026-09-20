import os
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import classification_report, confusion_matrix

TRAIN_DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "train_data.csv")
TEST_DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "test_data.csv")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

print("Loading train and test data...")
train_df = pd.read_csv(TRAIN_DATA_PATH)
test_df = pd.read_csv(TEST_DATA_PATH)

train_df['normalized_text'] = train_df['normalized_text'].fillna('')
test_df['normalized_text'] = test_df['normalized_text'].fillna('')

X_train = train_df['normalized_text']
X_test = test_df['normalized_text']

y_sentiment_train = train_df['sentiment']
y_sentiment_test = test_df['sentiment']

print("\nVectorizing text data (TF-IDF)...")
tfidf = TfidfVectorizer(ngram_range=(1, 2), max_features=5000)

X_train_vec = tfidf.fit_transform(X_train)
X_test_vec = tfidf.transform(X_test)

print(f"TF-IDF feature matrix shape (Train): {X_train_vec.shape}")

print("\nTraining Sentiment Classifier (Logistic Regression)...")
sentiment_model = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
sentiment_model.fit(X_train_vec, y_sentiment_train)

y_sentiment_pred = sentiment_model.predict(X_test_vec)
print("\n--- Sentiment Model Evaluation ---")
print(classification_report(y_sentiment_test, y_sentiment_pred))

print("\nPreparing Aspect Multi-Label Data...")
def parse_aspects(aspect_str):
    if pd.isna(aspect_str): return []
    return [a.strip() for a in str(aspect_str).split(",")]

y_aspect_train = train_df['aspects'].apply(parse_aspects)
y_aspect_test = test_df['aspects'].apply(parse_aspects)

mlb = MultiLabelBinarizer()
y_aspect_train_bin = mlb.fit_transform(y_aspect_train)
y_aspect_test_bin = mlb.transform(y_aspect_test)

print("Training Aspect Classifier (OneVsRest + Logistic Regression)...")
aspect_model = OneVsRestClassifier(LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42))
aspect_model.fit(X_train_vec, y_aspect_train_bin)

y_aspect_pred = aspect_model.predict(X_test_vec)
print("\n--- Aspect Model Evaluation ---")
print(classification_report(y_aspect_test_bin, y_aspect_pred, target_names=mlb.classes_))

print("\nSaving model artifacts to 'models/' directory...")
os.makedirs(MODEL_DIR, exist_ok=True)

joblib.dump(tfidf, os.path.join(MODEL_DIR, "tfidf_vectorizer.joblib"))
joblib.dump(sentiment_model, os.path.join(MODEL_DIR, "sentiment_model.joblib"))
joblib.dump(mlb, os.path.join(MODEL_DIR, "aspect_mlb.joblib"))
joblib.dump(aspect_model, os.path.join(MODEL_DIR, "aspect_model.joblib"))

print("Phase 3 Complete! Models exported successfully.")
