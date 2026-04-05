from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# load trained model
model_path = "model"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

labels = ["negative", "neutral", "positive"]

def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)

    with torch.no_grad():
        outputs = model(**inputs)

    scores = outputs.logits
    predicted_class = torch.argmax(scores).item()

    return labels[predicted_class]


while True:
    text = input("Enter Hinglish text: ")

    if text.lower() == "exit":
        break

    result = predict_sentiment(text)

    print("Sentiment:", result)
    print()
