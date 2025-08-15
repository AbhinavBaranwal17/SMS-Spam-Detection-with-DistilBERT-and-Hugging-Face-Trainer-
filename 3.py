import pandas as pd
import torch
import numpy as np
from datasets import Dataset
import transformers
from transformers import TrainingArguments, Trainer, DistilBertTokenizerFast, DistilBertForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

df = pd.read_csv(r"C:\Users\kenka\Desktop\Everything\python\files csv\SMSSpamCollection.txt", sep='\t', names=["label", "message"])

x = list(df["message"])

def get_spam(n):
    return 1 if n == 'spam' else 0

df['Spam'] = df['label'].apply(get_spam)
y = list(df['Spam'])

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=0)

tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
train_encodings = tokenizer(X_train, truncation=True, padding=True)
test_encodings = tokenizer(X_test, truncation=True, padding=True)

train_dataset = Dataset.from_dict({
    'input_ids': train_encodings['input_ids'],
    'attention_mask': train_encodings['attention_mask'],
    'labels': torch.tensor(y_train)
})

test_dataset = Dataset.from_dict({
    'input_ids': test_encodings['input_ids'],
    'attention_mask': test_encodings['attention_mask'],
    'labels': torch.tensor(y_test)
})

model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)

training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=24,
    per_device_eval_batch_size=64,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_dir='./logs',
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy"
)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics
)

trainer.train()

trainer.save_model("saved_spam_model")
tokenizer.save_pretrained("saved_spam_model")
print("\nModel and tokenizer saved to 'saved_spam_model/'\n")

from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast

model = DistilBertForSequenceClassification.from_pretrained("saved_spam_model")
tokenizer = DistilBertTokenizerFast.from_pretrained("saved_spam_model")

new_texts = [
    "Congratulations! You've won a $1000 Walmart gift card. Click here to claim.",
    "Hey, are we still meeting tomorrow?",
    "Your package has been delivered. Thank you for shopping with us!",
    "FREE entry into a weekly contest just by replying YES!"
]

new_encodings = tokenizer(new_texts, truncation=True, padding=True, return_tensors="pt")

model.eval()
with torch.no_grad():
    outputs = model(**new_encodings)
    logits = outputs.logits
    predictions = torch.argmax(logits, dim=1)

print("Predictions on new messages:\n")
for text, pred in zip(new_texts, predictions):
    label = "spam" if pred == 1 else "ham"
    print(f"{label.upper()} → {text}")



