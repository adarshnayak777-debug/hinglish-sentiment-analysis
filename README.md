# Hinglish Sentiment Analysis

A sentiment analysis system built for Hinglish text (a mix of Hindi and English). It classifies text as **positive** or **negative** using a transformer-based NLP model.

---

## What is Hinglish?

Hinglish is an informal blend of Hindi and English commonly used in social media, chats, and everyday conversations in India. Traditional NLP models struggle with this type of text, so this project focuses specifically on handling it.

---

## Project Structure

```
hinglish-sentiment-analysis/
│
├── raw_posts.json           # Original dataset
├── enriched_posts.json      # Expanded and enriched dataset
├── prepare_dataset.py       # Script to preprocess data and create CSV
├── train.py                 # Script to train the model
├── predict.py               # Script to run predictions
└── .gitignore
```

---

## Dataset

- The dataset was manually created in JSON format
- It contains Hinglish sentences from everyday conversations and social media
- Two versions are available:
  - `raw_posts.json` — original collected posts
  - `enriched_posts.json` — expanded version with more diverse examples
- Labels are assigned based on engagement score:
  - `engagement > 500` → **positive**
  - `engagement > 150` → **neutral**
  - `engagement <= 150` → **negative**
- Neutral labels are removed during training (binary classification only)

---

## Model Used

- **Model:** `bert-base-multilingual-cased`
- **Framework:** HuggingFace Transformers
- **Type:** Sequence Classification (2 labels — positive, negative)
- **Why this model?** It supports multiple languages including Hindi and English, making it suitable for Hinglish text

---

## How to Run

### 1. Clone the repository

```bash
git clone https://github.com/adarshnayak777-debug/hinglish-sentiment-analysis.git
cd hinglish-sentiment-analysis
```

### 2. Install dependencies

```bash
pip install transformers datasets pandas scikit-learn torch
```

### 3. Prepare the dataset

```bash
python prepare_dataset.py
```

This will generate `hinglish_sentiment.csv` inside a `dataset/` folder.

### 4. Train the model

```bash
python train.py
```

This will train the model and save it in the `model/` folder.

### 5. Run predictions

```bash
python predict.py
```

Enter any Hinglish text and it will predict the sentiment.

---

## Example

```
Enter Hinglish text: yaar job mil gayi finally!
Sentiment: positive
```

---

## Challenges

- No standard Hinglish dataset available publicly
- High variability in language usage and spelling
- Limited computational resources (CPU-only training)
- Labeling ambiguous sentences was difficult

---

## Future Improvements

- Add more data to improve accuracy
- Build a simple web interface for real-time predictions
- Try other multilingual models like IndicBERT or MuRIL
- Add neutral sentiment back as a third class

---

## Tech Stack

- Python
- HuggingFace Transformers
- PyTorch
- Pandas
- Scikit-learn
