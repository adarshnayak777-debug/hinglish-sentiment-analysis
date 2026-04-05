import json
import pandas as pd

with open("raw_posts.json", "r", encoding="utf-8") as f:
    data = json.load(f)

texts = []
labels = []

for post in data:
    text = post["text"]
    engagement = post["engagement"]

    if engagement > 500:
        label = "positive"
    elif engagement > 150:
        label = "neutral"
    else:
        label = "negative"

    texts.append(text)
    labels.append(label)

df = pd.DataFrame({
    "text": texts,
    "label": labels
})

print(df['label'].value_counts())

df.to_csv("hinglish_sentiment.csv", index=False)

print("Dataset created successfully")
