# NLP Job Matcher API

A FastAPI-powered REST API that matches resumes to job postings 
using TF-IDF vectorization and cosine similarity.

## 🚀 Live Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | / | Health check + total jobs count |
| POST | /match | Match resume to top-N job postings |

## 📊 Model
- Algorithm: TF-IDF + Cosine Similarity
- Dataset: 123,842 LinkedIn job postings
- Features: 5,000 TF-IDF features

## 📥 How to Use

Send a POST request to /match:

```json
{
  "resume_text": "Python machine learning data science",
  "top_n": 5
}
```
## 🛠 Setup
```bash
pip install -r requirements.txt
uvicorn main:app --reload
```

## 👤 Author
Bright Alfred Riochi | AI/ML Engineer
[GitHub](https://github.com/Riochi511) | [LinkedIn](https://www.linkedin.com/in/riochi-ai453b9)
