import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
from sklearn.model_selection import train_test_split

# 1. Load dataset
df = pd.read_csv("dataset/hinglish_sentiment.csv")

# 2. Basic cleaning
df['text'] = df['text'].astype(str).str.lower().str.strip()

# 3. Encode labels (now 3 classes)
label_map = {"negative": 0, "neutral": 1, "positive": 2}
df['label'] = df['label'].map(label_map)

# 4. Drop rows where label is missing
df = df.dropna(subset=['label'])
df['label'] = df['label'].astype(int)

# 5. Debug check
print("Label distribution:")
print(df['label'].value_counts())

# 6. Train-test split
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df['text'], df['label'], test_size=0.2, random_state=42
)

train_df = pd.DataFrame({"text": train_texts, "label": train_labels})
val_df = pd.DataFrame({"text": val_texts, "label": val_labels})

train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)

# 7. Load IndicBERT tokenizer
model_name = "ai4bharat/indic-bert"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize(example):
    return tokenizer(
        example["text"],
        padding="max_length",
        truncation=True,
        max_length=64
    )

train_dataset = train_dataset.map(tokenize, batched=True)
val_dataset = val_dataset.map(tokenize, batched=True)

# 8. Load IndicBERT model with 3 output labels
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=3
)

# 9. Training arguments - increased epochs for better accuracy
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=7,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    learning_rate=2e-5,
    warmup_steps=100,
    weight_decay=0.01,
    logging_dir="./logs",
    save_strategy="epoch",
)

# 10. Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

# 11. Train
trainer.train()

# 12. Save model
model.save_pretrained("model")
tokenizer.save_pretrained("model")

print("Training complete")