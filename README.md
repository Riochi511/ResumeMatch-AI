# 📄 NLP Job Matcher

An NLP-powered job matching application that analyzes a candidate's resume and ranks the most relevant job opportunities using TF-IDF vectorization and cosine similarity.

The application extracts text from uploaded resumes (PDF or DOCX), converts the resume into numerical feature vectors, compares it with job descriptions, and returns the top matching jobs through a FastAPI backend.

---

## 🚀 Project Overview
<img width="1600" height="663" alt="image (8)" src="https://github.com/user-attachments/assets/b3052123-6d13-4cea-a6b8-de54fcb03bf6" />
<img width="1600" height="632" alt="image (9)" src="https://github.com/user-attachments/assets/8a014795-115c-4723-938f-69ad5f94ada1" />
<img width="1600" height="670" alt="image (10)" src="https://github.com/user-attachments/assets/2abf19c9-01b4-4417-bd14-ef9daf0dda52" />
Finding the right job is difficult for many job seekers, especially when they have to manually search through hundreds of job postings.

This project automates that process by matching resumes with job descriptions based on textual similarity, helping applicants quickly identify jobs that best align with their skills and experience.

---

## 🎯 Problem Statement

Job seekers often spend significant time searching for suitable positions, while recruiters also need efficient ways to identify qualified candidates.

This project demonstrates how Natural Language Processing (NLP) can improve job matching by comparing resumes with job descriptions.

---

## 💡 Solution

The application:

- Accepts PDF and DOCX resumes.
- Extracts resume text.
- Converts text into TF-IDF vectors.
- Compares the resume against stored job description vectors using cosine similarity.
- Returns the Top 10 most relevant job matches with similarity scores.

---

## 🏗️ System Workflow

```
Resume Upload
       │
       ▼
Extract Text (PDF/DOCX)
       │
       ▼
TF-IDF Vectorizer (transform)
       │
       ▼
Resume Vector
       │
       ▼
Cosine Similarity
       │
       ▼
Compare Against All Job Vectors
       │
       ▼
Rank Results
       │
       ▼
Top 10 Job Matches
```

---

## ⚙️ Tech Stack

- Python
- FastAPI
- Scikit-learn
- Pandas
- NumPy
- Joblib
- pdfplumber
- python-docx

---

## 🧠 NLP Techniques

### TF-IDF (Term Frequency–Inverse Document Frequency)

TF-IDF converts text into numerical vectors while assigning higher importance to informative words and reducing the importance of common words.

### Cosine Similarity

Cosine similarity measures how similar the resume vector is to each job description vector. Jobs with the highest similarity scores are ranked first.

---

## Why TF-IDF?

TF-IDF was chosen because it assigns higher importance to informative words while reducing the influence of very common words.

For example, words like "Python", "TensorFlow", and "Machine Learning" contribute more to matching than common words such as "the", "and", or "experience".

This makes resume-job matching more meaningful while remaining lightweight and computationally efficient.

---

## Why Cosine Similarity?

Cosine similarity measures how similar two TF-IDF vectors are.

Instead of counting common words, it compares the direction of the vectors.

A higher cosine similarity score indicates that the resume is more closely aligned with the job description.

---

## 📂 Project Structure

```
nlp-job-matcher/
│
├── main.py
├── vectorizer.pkl
├── tfidf_matrix.pkl
├── jobs.csv
├── requirements.txt
└── README.md
```

---

## ▶️ Running the Project

Clone the repository

```bash
git clone https://github.com/Riochi511/nlp-job-matcher.git
```

Install dependencies

```bash
pip install -r requirements.txt
```

Start the API

```bash
uvicorn main:app --reload
```

Visit:

```
http://127.0.0.1:8000/docs
```

to test the API using FastAPI's interactive Swagger interface.

---

## 🎯 Business Value

This solution can help:

- Job seekers discover relevant opportunities faster.
- Recruitment platforms improve job recommendations.
- Career guidance systems personalize job suggestions.
- HR teams build intelligent resume screening workflows.

---

## 🔮 Future Improvements

- Replace TF-IDF with Sentence Transformers for semantic matching.
- Explain why each job was recommended by highlighting matched skills.
- Include years of experience, education, certifications, and location in the ranking process.
- Add authentication and database support.
- Deploy the API to a cloud platform.
- Add automated tests and CI/CD.

---
## Lessons Learned

Through this project I learned:

- How TF-IDF converts text into numerical vectors.
- Why transform() is used during inference instead of fit_transform().
- How cosine similarity ranks resumes against thousands of jobs.
- How to build and deploy an NLP application using FastAPI.
- How machine learning models can solve real-world recruitment problems.
  
  ---
  
## 👨‍💻 Author

**Alfred Bright Riochi**

AI / Machine Learning Engineer

GitHub:
https://github.com/Riochi511

LinkedIn:
https://www.linkedin.com/in/riochi-ai453b9
