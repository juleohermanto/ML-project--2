import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score

print("Loading datasets...")
train_df = pd.read_csv("train_data.csv").fillna('')
test_df = pd.read_csv("test_data.csv").fillna('')

X_train = train_df['normalized_text'].tolist()
X_test = test_df['normalized_text'].tolist()

label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(train_df['sentiment'])
y_test = label_encoder.transform(test_df['sentiment'])

print("Downloading IndoBERT Tokenizer...")
model_name = "indobenchmark/indobert-base-p1"
tokenizer = AutoTokenizer.from_pretrained(model_name)

train_encodings = tokenizer(X_train, truncation=True, padding=True, max_length=128)
test_encodings = tokenizer(X_test, truncation=True, padding=True, max_length=128)

class ReviewDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = ReviewDataset(train_encodings, y_train)
test_dataset = ReviewDataset(test_encodings, y_test)

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average='macro')
    return {'accuracy': acc, 'f1': f1}

print("Downloading IndoBERT Model...")
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)

training_args = TrainingArguments(
    output_dir='./results',          
    num_train_epochs=3,              # 3 Passes over the data
    per_device_train_batch_size=16,  
    per_device_eval_batch_size=16,   
    evaluation_strategy="epoch",     
    save_strategy="epoch",
    logging_dir='./logs',            
    logging_steps=10,
    load_best_model_at_end=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics
)

print("Starting Training on GPU...")
trainer.train()

print("Saving model...")
trainer.save_model("./indobert_sentiment_model")
tokenizer.save_pretrained("./indobert_sentiment_model")

print("Training Complete! Download the 'indobert_sentiment_model' folder from Colab to your local machine.")
