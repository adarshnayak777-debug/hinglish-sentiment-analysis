# Hinglish Sentiment Analysis

A sentiment analysis system built to classify Hinglish (Hindi + English mixed) text into **Positive**, **Negative**, and **Neutral** sentiments using the IndicBERT transformer model.

---

## What is Hinglish?

Hinglish is a blend of Hindi and English commonly used in everyday conversations, social media, chats, and comments in India. Example: *"yaar ye phone bahut acha hai!"* or *"I am so happy aaj!"*

---

## Project Structure

```
hinglish-sentiment-analysis/
│
├── dataset/
│   └── hinglish_sentiment.csv    # 3500+ labeled Hinglish sentences
│
├── model/                         # Trained IndicBERT model weights
├── results/                       # Training checkpoints
├── app/                           # Web app (coming soon)
│
├── raw_posts.json                 # Raw social media posts
├── enriched_posts.json            # Enriched post data
├── prepare_dataset.py             # Converts JSON to CSV dataset
├── train.py                       # Model training script
├── predict.py                     # Run predictions on text input
└── requirements.txt
```

---

## Dataset

Custom dataset created manually with **3500+ Hinglish sentences** covering:

- Workplace situations
- Student life
- Relationships
- Social media and internet slang
- Mixed English-Hindi expressions

| Label | Count |
|-------|-------|
| Positive | ~1100 |
| Negative | ~1200 |
| Neutral | ~1200 |

---

## Model

Uses **IndicBERT** (`ai4bharat/indic-bert`) — a multilingual ALBERT model pretrained on 12 Indian languages including Hindi. Fine-tuned on the custom Hinglish dataset for 3-class sentiment classification.

| Parameter | Value |
|-----------|-------|
| Model | ai4bharat/indic-bert |
| Epochs | 7 |
| Batch Size | 8 |
| Learning Rate | 2e-5 |
| Max Sequence Length | 64 |

---

## Setup

**1. Clone the repo**
```bash
git clone https://github.com/your-username/hinglish-sentiment-analysis.git
cd hinglish-sentiment-analysis
```

**2. Create virtual environment**
```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux
```

**3. Install dependencies**
```bash
pip install transformers datasets scikit-learn pandas torch
pip install protobuf tiktoken
```

**4. Login to HuggingFace** (required for IndicBERT)
```bash
python -c "from huggingface_hub import login; login(token='your_hf_token')"
```

> Request access to IndicBERT at: https://huggingface.co/ai4bharat/indic-bert

---

## Training

```bash
python prepare_dataset.py    # prepare the dataset first
python train.py              # train the model (takes 2-4 hours on CPU)
```

---

## Prediction

```bash
python predict.py
```

Example output:
```
Enter Hinglish text: mai khush hoon
Sentiment: positive

Enter Hinglish text: hatt! pagal
Sentiment: negative

Enter Hinglish text: aaj office mein meeting thi
Sentiment: neutral
```

---

## Challenges

- No standard Hinglish dataset available — built from scratch
- IndicBERT is a gated model — requires HuggingFace authentication
- CPU-only training — each run takes 3-4 hours
- High language variability — slang, abbreviations, mixed scripts

---

## Future Work

- Build a web interface for real-time predictions
- Train on GPU for faster and more epochs
- Expand dataset further
- Support more Indian language code-switching patterns

---

## References

- Kakwani et al. (2020). IndicNLPSuite — https://huggingface.co/ai4bharat/indic-bert
- Devlin et al. (2019). BERT: Pre-training of Deep Bidirectional Transformers
- HuggingFace Transformers — https://huggingface.co/docs/transformers

---

## Author

Made by Adarsh.
