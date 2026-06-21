from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
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

HTML_PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>NLP Job Matcher</title>
<style>
  * { box-sizing: border-box; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    background: #0f1117;
    color: #e6e6e6;
    margin: 0;
    padding: 40px 20px;
    display: flex;
    justify-content: center;
  }
  .container { max-width: 700px; width: 100%; }
  h1 { font-size: 1.8rem; margin-bottom: 4px; }
  .subtitle { color: #9aa0ac; margin-bottom: 28px; font-size: 0.95rem; }
  textarea {
    width: 100%;
    min-height: 160px;
    background: #1a1d27;
    border: 1px solid #2c2f3a;
    border-radius: 10px;
    color: #e6e6e6;
    padding: 14px;
    font-size: 0.95rem;
    resize: vertical;
    font-family: inherit;
  }
  textarea:focus { outline: none; border-color: #5b8def; }
  .controls {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-top: 14px;
  }
  .controls label { font-size: 0.85rem; color: #9aa0ac; }
  .controls input[type=number] {
    width: 60px;
    background: #1a1d27;
    border: 1px solid #2c2f3a;
    border-radius: 6px;
    color: #e6e6e6;
    padding: 6px 8px;
    margin-left: 8px;
  }
  button {
    background: #5b8def;
    color: white;
    border: none;
    border-radius: 8px;
    padding: 10px 22px;
    font-size: 0.95rem;
    font-weight: 600;
    cursor: pointer;
  }
  button:hover { background: #4a7adb; }
  button:disabled { background: #3a4050; cursor: not-allowed; }
  #results { margin-top: 28px; }
  .match {
    background: #1a1d27;
    border: 1px solid #2c2f3a;
    border-radius: 10px;
    padding: 14px 16px;
    margin-bottom: 10px;
  }
  .match-title { font-weight: 600; font-size: 1rem; }
  .match-company { color: #9aa0ac; font-size: 0.85rem; margin-top: 2px; }
  .match-score {
    display: inline-block;
    margin-top: 6px;
    background: #233047;
    color: #7fb3ff;
    font-size: 0.75rem;
    padding: 2px 8px;
    border-radius: 6px;
  }
  .status { color: #9aa0ac; font-size: 0.85rem; margin-top: 10px; }
  .footer { margin-top: 40px; font-size: 0.8rem; color: #5a5f6b; }
  .footer a { color: #7fb3ff; }
</style>
</head>
<body>
<div class="container">
  <h1>NLP Job Matcher</h1>
  <div class="subtitle">Paste your resume below to find matching jobs from 123,842 real LinkedIn postings.</div>

  <textarea id="resumeText" placeholder="Paste your resume or a summary of your skills and experience..."></textarea>

  <div class="controls">
    <div><label>Number of matches <input type="number" id="topN" value="5" min="1" max="50"></label></div>
    <button id="submitBtn" onclick="findMatches()">Find Matches</button>
  </div>

  <div id="status" class="status"></div>
  <div id="results"></div>

  <div class="footer">
    Built by Bright Alfred Riochi &middot;
    <a href="https://github.com/Riochi511/nlp-job-matcher" target="_blank">View source on GitHub</a> &middot;
    <a href="/docs" target="_blank">API docs</a>
  </div>
</div>

<script>
async function findMatches() {
  const resumeText = document.getElementById('resumeText').value.trim();
  const topN = parseInt(document.getElementById('topN').value) || 5;
  const status = document.getElementById('status');
  const results = document.getElementById('results');
  const btn = document.getElementById('submitBtn');

  if (!resumeText) {
    status.textContent = "Please paste some resume text first.";
    return;
  }

  btn.disabled = true;
  btn.textContent = "Searching...";
  status.textContent = "This may take up to a minute on first request (cold start).";
  results.innerHTML = "";

  try {
    const res = await fetch('/match', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ resume_text: resumeText, top_n: topN })
    });

    if (!res.ok) throw new Error("Request failed");

    const data = await res.json();
    status.textContent = `Found ${data.matches.length} matches.`;

    results.innerHTML = data.matches.map(m => `
      <div class="match">
        <div class="match-title">${m.title}</div>
        <div class="match-company">${m.company}</div>
        <div class="match-score">Similarity: ${(m.similarity_score * 100).toFixed(1)}%</div>
      </div>
    `).join('');
  } catch (err) {
    status.textContent = "Something went wrong. Please try again.";
  } finally {
    btn.disabled = false;
    btn.textContent = "Find Matches";
  }
}
</script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
def home():
    return HTML_PAGE

@app.get("/health")
def health():
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