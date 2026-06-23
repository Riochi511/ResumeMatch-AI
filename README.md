# NLP Job Matcher

Match your resume to the most relevant job postings using NLP — no sign-up, no setup.

**Live demo:** [huggingface.co/spaces/Riochi511/nlp-job-matcher](https://huggingface.co/spaces/Riochi511/nlp-job-matcher)

---

## What it does

Paste your resume or upload a PDF/DOCX file. The app vectorises your text using TF-IDF and ranks 124,000+ LinkedIn job postings by cosine similarity — returning the top matches with title, company, a description preview, and a match score.

---

## How it works

1. Resume text is transformed into a TF-IDF vector using a pre-trained `TfidfVectorizer`
2. Cosine similarity is computed against a pre-built matrix of 124K+ job descriptions
3. The top N results are returned, ranked by similarity score

---

## Tech stack

| Layer | Tools |
|---|---|
| NLP model | TF-IDF · Cosine Similarity (scikit-learn) |
| API | FastAPI · Uvicorn |
| File parsing | pdfplumber · python-docx |
| Dataset | LinkedIn Job Postings (~124K rows) |
| Deployment | Docker · Hugging Face Spaces |

---

## Dataset

[LinkedIn Job Postings — Kaggle](https://www.kaggle.com/datasets/arshkon/linkedin-job-postings)

Cleaned to 124K rows with columns: `job_id`, `title`, `company_name`, `description`.

---

## API

### `POST /match`

| Field | Type | Description |
|---|---|---|
| `resume_text` | string (form) | Plain text resume |
| `resume_file` | file upload | PDF or DOCX |
| `top_n` | int (form) | Number of results (default: 10) |

**Example response:**
```json
{
  "matches": [
    {
      "rank": 1,
      "title": "Applied Machine Learning Research Scientist",
      "company": "Coactive AI",
      "preview": "We are looking for a researcher with experience in NLP...",
      "similarity": 0.4742
    }
  ],
  "total": 10
}
```

---

## Run locally

```bash
git clone https://github.com/Riochi511/nlp-job-matcher
cd nlp-job-matcher
pip install -r requirements.txt
uvicorn main:app --reload
```

Open [http://localhost:8000](http://localhost:8000)

> The app downloads `tfidf_vectorizer.pkl`, `tfidf_matrix.pkl`, and `jobs_clean.csv` from Google Drive on first run.

---

## Project structure

```
nlp-job-matcher/
├── main.py                 # FastAPI app + frontend UI
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## Related projects

- [Beijing AQI Forecaster](https://github.com/Riochi511/aqi-forecaster) — time series forecasting with Facebook Prophet
-