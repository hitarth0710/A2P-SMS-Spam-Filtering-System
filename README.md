# A2P-SMS-Spam-Filtering-System

## Project Overview

A lightweight AI-based spam filtering system for A2P (Application-to-Person) SMS messages. The system classifies messages into **Transactional**, **Promotional**, and **Spam** categories, incorporates a whitelisting mechanism for trusted domains and phrases to minimize false positives, and exposes filtering logic via a REST API.

---

## Video Explanation

Link: *[Add your video link here]*

---

## Features

- **Multi-class SMS classification**: Transactional, Promotional, Spam
- **Whitelist support**: Trusted domains and phrases bypass filtering
- **REST API**: Real-time message classification
- **Configurable**: Easily update whitelist entries via YAML
- **Logging**: Tracks API requests, predictions, and errors
- **Docker-ready**: Containerized for easy deployment
- **Unit & API Testing**: Automated test suite for reliability

---

## Quick Start

### 1. Clone the Repository

```bash
git clone <your_repository_url>
cd A2P-SMS-Spam-Filtering-System
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Prepare Data & Train Model

```bash
python main.py --train
```
- Preprocesses `data/raw/message_dataset_50k.csv`
- Trains a Multinomial Naive Bayes classifier (TF-IDF features)
- Saves model to `data/models/model.pkl`

### 4. Run the API Server

```bash
python main.py --api --host 0.0.0.0 --port 8000
```

Or with Docker:

```bash
docker build -t sms-filter .
docker run -p 8000:8000 sms-filter
```

---

## Configuration

### Whitelist Entries

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

*Changes take effect on API restart.*

---

## API Usage

**Endpoint:** `POST /check_sms`

**Request Example:**
```json
{
  "message": "Your OTP is 123456. Do not share it."
}
```

**Response Example:**
```json
{
  "verdict": "allowed",
  "reason": "whitelisted",
  "category": "Transactional"
}
```

**Curl Example:**
```bash
curl -X POST http://localhost:8000/check_sms \
  -H "Content-Type: application/json" \
  -d '{"message": "Visit our site for amazing deals!"}'
```

**Response Fields:**
- `verdict`: `"allowed"` or `"blocked"`
- `reason`: `"whitelisted"` or `"ai"`
- `category`: `"Transactional"`, `"Promotional"`, or `"Spam"`

**Health Check:**  
`GET /health` returns API status.

---

## Project Structure

```
A2P-SMS-Spam-Filtering-System/
├── api/
│   └── app.py
├── config/
│   └── config.yaml
├── data/
│   ├── models/
│   │   └── model.pkl
│   ├── processed/
│   │   └── labeled_messages.csv
│   └── raw/
│       └── message_dataset_50k.csv
├── logs/
│   ├── app.log
│   ├── filter.log
├── notebooks/
│   └── spam.ipynb
├── src/
│   ├── data/
│   │   └── preprocessing.py
│   ├── models/
│   │   └── sms_filter.py
│   └── utils/
│       └── whitelist.py
├── tests/
│   ├── results/
│   │   └── evaluation_results.txt
│   ├── test_api.py
│   ├── test_model.py
├── Dockerfile
├── main.py
├── requirements.txt
├── run.log
└── request.json
```

---

## Testing

- All tests are in the `tests/` folder.
- Run tests with:
  ```bash
  pytest tests/
  ```
- Test results and evaluation metrics are saved in `tests/results/evaluation_results.txt`.

---

## Logging

- **API logs**: `logs/app.log`
- **Model/classification logs**: `logs/filter.log`
- **Run logs**: `run.log`
- Logs include request details, predictions, whitelist matches, and errors.

---

## Dataset

- **Raw data**: `data/raw/message_dataset_50k.csv`
- **Processed/labeled data**: `data/processed/labeled_messages.csv`
- **Model file**: `data/models/model.pkl`

---

## Notebook

- [`notebooks/spam.ipynb`](notebooks/spam.ipynb):  
  Contains data exploration, preprocessing, model training, and evaluation steps.

---

## Dependencies

Main dependencies:
- Flask
- scikit-learn
- pandas
- joblib
- pyyaml
- pytest

Install with:
```bash
pip install -r requirements.txt
```

---

## Docker Deployment

Build and run the container:
```bash
docker build -t sms-filter .
docker run -p 8000:8000 sms-filter
```

---

## Contribution

Feel free to open issues or submit pull requests for improvements, bug fixes, or new features.

---

## License

*Specify your license here (e.g., MIT, Apache 2.0, etc.)*

---

## Contact

For questions or support, contact: *[your-email@example.com]*

---
