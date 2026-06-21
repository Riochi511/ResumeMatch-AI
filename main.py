from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import scipy.sparse

app = FastAPI()

# Load artifacts at startup
vectorizer = joblib.load("tfidf_vectorizer.pkl")
jobs_df = pd.read_csv("jobs_clean.csv")
tfidf_matrix = vectorizer.transform(jobs_df["description"].fillna(""))

class ResumeInput(BaseModel):
    resume_text: str
    top_n: int = 10

@app.get("/")
def home():
    return {"message": "NLP Job Matcher API is live", "total_jobs": len(jobs_df)}

@app.post("/match")
def match_jobs(input: ResumeInput):
    top_n = max(1, min(input.top_n, 50))
    
    resume_vector = vectorizer.transform([input.resume_text])
    similarities = cosine_similarity(resume_vector, tfidf_matrix).flatten()
    top_indices = similarities.argsort()[::-1][:top_n]
    
    results = []
    for idx in top_indices:
        row = jobs_df.iloc[idx]
        results.append({
            "title": row["title"],
            "company": row["company_name"] if pd.notna(row["company_name"]) else "Unknown",
            "similarity_score": round(float(similarities[idx]), 4)
        })
    
    return {"matches": results}