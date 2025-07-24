# A2P-SMS-Spam-Filtering-System

## Project Overview

A lightweight AI-based spam filtering system for A2P (Application-to-Person) SMS messages that classifies messages into **Transactional**, **Promotional**, and **Spam** categories. The system incorporates a whitelisting mechanism for trusted domains and phrases to minimize false positives and exposes filtering logic via a REST API.

## Video Explanation 
Link :- https://www.loom.com/share/86c081c48d7147d481fd684caac6251d?sid=93456197-9f5a-4977-8f30-38d596c33168

## Quick Start

### Training the Model

1. **Preprocess and train:**
```bash
python main.py --train
```

This will:
- Preprocess the dataset from `data/raw/message_dataset_50k.csv`
- Train a Multinomial Naive Bayes classifier with TF-IDF vectorization
- Save the trained model to `data/models/model.pkl`

### Running the API

```bash
python main.py --api --host 0.0.0.0 --port 8000
```

Or using Docker:
```bash
docker build -t sms-filter .
docker run -p 8000:8000 sms-filter
```

## Adding Whitelist Entries

Edit [`config/config.yaml`](config/config.yaml):

```yaml
whitelisted_domains:
  - trip.com
  - icicibank.com
  - yourbank.com

whitelisted_phrases:
  - "Your OTP is"
  - "Thank you for shopping with"
  - "Order confirmation"
```

The system automatically reloads configuration on startup.

## API Usage

**Endpoint:** `POST /check_sms`

**Request:**
```json
{
  "message": "Your OTP is 123456. Do not share it."
}
```

**Response:**
```json
{
  "verdict": "allowed",
  "reason": "whitelisted",
  "category": "Transactional"
}
```

**Example with curl:**
```bash
curl -X POST http://localhost:8000/check_sms \
  -H "Content-Type: application/json" \
  -d '{"message": "Visit our site for amazing deals!"}'
```

**Response Fields:**
- `verdict`: `"allowed"` or `"blocked"`
- `reason`: `"whitelisted"` or `"ai"`
- `category`: `"Transactional"`, `"Promotional"`, or `"Spam"`

## Project Structure

```
A2P-SMS-Spam-Filtering-System/
├── api/
│   └── app.py                          # Flask REST API server
├── config/
│   └── config.yaml                     # Whitelist configuration (domains & phrases)
├── data/
│   ├── models/
│   │   └── model.pkl                   # Trained ML model (auto-generated)
│   ├── processed/
│   │   └── labeled_messages.csv        # Preprocessed dataset (auto-generated)
│   └── raw/
│       └── message_dataset_50k.csv     # Original 50K SMS dataset with labels
├── logs/
│   ├── app.log                         # Application runtime logs
│   ├── filter.log                      # Legacy application logs (auto-generated)
├── notebooks/
│   └── spam.ipynb                      # Jupyter notebook for development/analysis
├── src/
│   ├── data/
│   │   └── preprocessing.py            # Data cleaning & feature extraction
│   ├── models/
│   │   └── sms_filter.py              # Core SMS classification service
│   └── utils/
│       └── whitelist.py               # Whitelist management utilities
├── tests/
│   ├── results/                        # Containg report from testing
│   ├── test_api.py                     # API endpoint tests
│   ├── test_model.py           # Model tests
├── Dockerfile                          # Container deployment configuration
├── main.py                            # Application entry point & CLI
├── requirements.txt                   # Python dependencies
└── run.log                            # Running Flask API
|__ request.json
```

## Dependencies

```bash
pip install -r requirements.txt
```

Main dependencies: Flask, scikit-learn, pandas, joblib, pyyaml

---

**Health Check:** `GET /health` - Returns API status
