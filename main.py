import io
import os
import joblib
import numpy as np
import pandas as pd
import pdfplumber
import docx
import gdown

from fastapi import FastAPI, Form, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
from sklearn.metrics.pairwise import cosine_similarity
from typing import Optional

# ── Download artifacts from Google Drive on cold start ───────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

ARTIFACTS = {
    "tfidf_vectorizer.pkl": "1Hxjw3hZOGdym32IfMKrnrUM4PZ3MlEOl",
    "tfidf_matrix.pkl":     "1ad1dQC9UXwipP5jhEr4jP1S_WAmqSLNm",
    "jobs_clean.csv":       "19DW0X-8bUV6S2nWayf5p1DsWp0m4NqqY",
}

for filename, file_id in ARTIFACTS.items():
    dest = os.path.join(BASE_DIR, filename)
    if not os.path.exists(dest):
        print(f"Downloading {filename} from Google Drive...")
        gdown.download(f"https://drive.google.com/uc?id={file_id}", dest, quiet=False)
    else:
        print(f"{filename} already exists, skipping download.")

# ── Load artifacts ────────────────────────────────────────────────────────────
vectorizer   = joblib.load(os.path.join(BASE_DIR, "tfidf_vectorizer.pkl"))
tfidf_matrix = joblib.load(os.path.join(BASE_DIR, "tfidf_matrix.pkl"))
jobs_df      = pd.read_csv(os.path.join(BASE_DIR, "jobs_clean.csv"))

app = FastAPI(title="NLP Job Matcher")

# ── HTML Frontend ─────────────────────────────────────────────────────────────
HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>NLP Job Matcher</title>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&family=DM+Serif+Display&display=swap" rel="stylesheet"/>
  <style>
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

    :root {
      --navy:    #0B1F3A;
      --slate:   #1E3A5F;
      --sky:     #2E6DA4;
      --ice:     #EAF2FB;
      --white:   #FFFFFF;
      --ink:     #1A1A2E;
      --muted:   #64748B;
      --border:  #CBD5E1;
      --green:   #0F7B55;
      --green-bg:#E6F4EF;
    }

    body {
      font-family: 'Inter', sans-serif;
      background: #F0F4F9;
      color: var(--ink);
      min-height: 100vh;
    }

    /* ── Header ── */
    header {
      background: var(--navy);
      padding: 0 2rem;
      display: flex;
      align-items: center;
      justify-content: space-between;
      height: 64px;
      border-bottom: 3px solid var(--sky);
    }
    .logo {
      font-family: 'DM Serif Display', serif;
      font-size: 1.35rem;
      color: var(--white);
      letter-spacing: 0.01em;
    }
    .logo span { color: #5FB3F5; }
    .badge {
      font-size: 0.7rem;
      font-weight: 600;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      color: #5FB3F5;
      border: 1px solid #2E6DA4;
      padding: 3px 10px;
      border-radius: 20px;
    }

    /* ── Hero ── */
    .hero {
      background: linear-gradient(135deg, var(--navy) 0%, var(--slate) 100%);
      color: var(--white);
      padding: 3.5rem 2rem 3rem;
      text-align: center;
    }
    .hero h1 {
      font-family: 'DM Serif Display', serif;
      font-size: clamp(1.8rem, 4vw, 2.8rem);
      font-weight: 400;
      line-height: 1.2;
      margin-bottom: 0.75rem;
    }
    .hero h1 em {
      font-style: normal;
      color: #5FB3F5;
    }
    .hero p {
      font-size: 1rem;
      color: #A8C4E0;
      max-width: 520px;
      margin: 0 auto;
      line-height: 1.6;
    }

    /* ── Main layout ── */
    .container {
      max-width: 820px;
      margin: 0 auto;
      padding: 2rem 1.5rem 4rem;
    }

    /* ── Card ── */
    .card {
      background: var(--white);
      border-radius: 12px;
      border: 1px solid var(--border);
      padding: 2rem;
      box-shadow: 0 1px 4px rgba(0,0,0,0.06);
    }

    .section-label {
      font-size: 0.7rem;
      font-weight: 600;
      letter-spacing: 0.1em;
      text-transform: uppercase;
      color: var(--sky);
      margin-bottom: 1.25rem;
    }

    /* ── Tabs ── */
    .tabs {
      display: flex;
      gap: 0;
      border: 1px solid var(--border);
      border-radius: 8px;
      overflow: hidden;
      margin-bottom: 1.5rem;
    }
    .tab-btn {
      flex: 1;
      padding: 0.65rem 1rem;
      border: none;
      background: #F8FAFC;
      color: var(--muted);
      font-size: 0.875rem;
      font-weight: 500;
      cursor: pointer;
      transition: all 0.15s;
      font-family: 'Inter', sans-serif;
    }
    .tab-btn:first-child { border-right: 1px solid var(--border); }
    .tab-btn.active {
      background: var(--navy);
      color: var(--white);
    }

    /* ── Textarea ── */
    textarea {
      width: 100%;
      height: 220px;
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 0.875rem 1rem;
      font-size: 0.875rem;
      font-family: 'Inter', sans-serif;
      color: var(--ink);
      resize: vertical;
      outline: none;
      transition: border-color 0.15s;
      line-height: 1.6;
    }
    textarea:focus { border-color: var(--sky); box-shadow: 0 0 0 3px rgba(46,109,164,0.1); }

    /* ── File upload ── */
    .file-zone {
      border: 2px dashed var(--border);
      border-radius: 8px;
      padding: 2.5rem 1rem;
      text-align: center;
      cursor: pointer;
      transition: all 0.15s;
      background: #FAFBFD;
    }
    .file-zone:hover, .file-zone.dragover {
      border-color: var(--sky);
      background: var(--ice);
    }
    .file-zone input { display: none; }
    .upload-icon {
      font-size: 2rem;
      margin-bottom: 0.5rem;
      display: block;
    }
    .file-zone p {
      font-size: 0.875rem;
      color: var(--muted);
      margin-bottom: 0.25rem;
    }
    .file-zone .file-types {
      font-size: 0.75rem;
      color: #94A3B8;
    }
    .file-name {
      margin-top: 0.75rem;
      font-size: 0.8rem;
      color: var(--green);
      font-weight: 500;
    }

    /* ── Top N ── */
    .options-row {
      display: flex;
      align-items: center;
      gap: 1rem;
      margin-top: 1.25rem;
    }
    .options-row label {
      font-size: 0.825rem;
      color: var(--muted);
      font-weight: 500;
      white-space: nowrap;
    }
    .options-row select {
      border: 1px solid var(--border);
      border-radius: 6px;
      padding: 0.4rem 0.75rem;
      font-size: 0.825rem;
      font-family: 'Inter', sans-serif;
      color: var(--ink);
      outline: none;
      background: white;
      cursor: pointer;
    }

    /* ── Submit button ── */
    .btn-submit {
      display: block;
      width: 100%;
      margin-top: 1.5rem;
      padding: 0.875rem;
      background: var(--navy);
      color: var(--white);
      border: none;
      border-radius: 8px;
      font-size: 0.95rem;
      font-weight: 600;
      font-family: 'Inter', sans-serif;
      cursor: pointer;
      letter-spacing: 0.02em;
      transition: background 0.15s;
    }
    .btn-submit:hover { background: var(--slate); }
    .btn-submit:disabled { background: #94A3B8; cursor: not-allowed; }

    /* ── Loading ── */
    .loading {
      display: none;
      text-align: center;
      padding: 2.5rem;
      color: var(--muted);
    }
    .spinner {
      width: 36px; height: 36px;
      border: 3px solid var(--border);
      border-top-color: var(--sky);
      border-radius: 50%;
      animation: spin 0.7s linear infinite;
      margin: 0 auto 1rem;
    }
    @keyframes spin { to { transform: rotate(360deg); } }

    /* ── Results ── */
    #results { display: none; margin-top: 2rem; }

    .results-header {
      display: flex;
      align-items: baseline;
      justify-content: space-between;
      margin-bottom: 1rem;
    }
    .results-header h2 {
      font-family: 'DM Serif Display', serif;
      font-size: 1.35rem;
      font-weight: 400;
      color: var(--navy);
    }
    .results-count {
      font-size: 0.8rem;
      color: var(--muted);
    }

    .job-card {
      background: var(--white);
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 1.25rem 1.5rem;
      margin-bottom: 0.75rem;
      display: grid;
      grid-template-columns: 36px 1fr auto;
      gap: 0 1rem;
      align-items: start;
      transition: box-shadow 0.15s;
    }
    .job-card:hover { box-shadow: 0 4px 12px rgba(0,0,0,0.08); }

    .rank-num {
      font-size: 0.7rem;
      font-weight: 700;
      color: var(--sky);
      background: var(--ice);
      border-radius: 6px;
      width: 36px; height: 36px;
      display: flex; align-items: center; justify-content: center;
      margin-top: 2px;
    }

    .job-info { min-width: 0; }
    .job-title {
      font-size: 0.95rem;
      font-weight: 600;
      color: var(--navy);
      margin-bottom: 0.2rem;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }
    .job-company {
      font-size: 0.8rem;
      color: var(--sky);
      font-weight: 500;
      margin-bottom: 0.5rem;
    }
    .job-preview {
      font-size: 0.8rem;
      color: var(--muted);
      line-height: 1.5;
    }

    .score-pill {
      font-size: 0.72rem;
      font-weight: 700;
      padding: 4px 10px;
      border-radius: 20px;
      white-space: nowrap;
      margin-top: 2px;
    }
    .score-high   { background: var(--green-bg); color: var(--green); }
    .score-medium { background: #FEF3C7; color: #92400E; }
    .score-low    { background: #F1F5F9; color: var(--muted); }

    /* ── Error ── */
    .error-box {
      display: none;
      background: #FEF2F2;
      border: 1px solid #FECACA;
      border-radius: 8px;
      padding: 1rem 1.25rem;
      margin-top: 1rem;
      font-size: 0.875rem;
      color: #991B1B;
    }

    /* ── Footer ── */
    footer {
      text-align: center;
      padding: 1.5rem;
      font-size: 0.75rem;
      color: #94A3B8;
      border-top: 1px solid var(--border);
    }
  </style>
</head>
<body>

<header>
  <div class="logo">Job<span>Match</span></div>
  <div class="badge">NLP · TF-IDF</div>
</header>

<div class="hero">
  <h1>Find jobs that match<br/><em>your experience</em></h1>
  <p>Paste your resume or upload a file. Our NLP model ranks the best-fit roles from 124,000+ LinkedIn postings.</p>
</div>

<div class="container">
  <div class="card">
    <div class="section-label">Your Resume</div>

    <div class="tabs">
      <button class="tab-btn active" onclick="switchTab('text')">Paste Text</button>
      <button class="tab-btn" onclick="switchTab('file')">Upload File</button>
    </div>

    <div id="tab-text">
      <textarea id="resume-text" placeholder="Paste your resume or a summary of your skills and experience here..."></textarea>
    </div>

    <div id="tab-file" style="display:none">
      <div class="file-zone" id="drop-zone" onclick="document.getElementById('file-input').click()">
        <input type="file" id="file-input" accept=".pdf,.docx" onchange="handleFile(this)"/>
        <span class="upload-icon">📄</span>
        <p><strong>Click to upload</strong> or drag and drop</p>
        <p class="file-types">PDF or DOCX · Max 10 MB</p>
        <div class="file-name" id="file-name"></div>
      </div>
    </div>

    <div class="options-row">
      <label for="top-n">Number of results</label>
      <select id="top-n">
        <option value="5">5 jobs</option>
        <option value="10" selected>10 jobs</option>
        <option value="20">20 jobs</option>
      </select>
    </div>

    <button class="btn-submit" id="submit-btn" onclick="submitResume()">Find Matching Jobs</button>
    <div class="error-box" id="error-box"></div>
  </div>

  <div class="loading" id="loading">
    <div class="spinner"></div>
    <p>Analysing your resume against 124K+ job postings…</p>
  </div>

  <div id="results">
    <div class="results-header">
      <h2>Top Matches</h2>
      <span class="results-count" id="results-count"></span>
    </div>
    <div id="results-list"></div>
  </div>
</div>

<footer>Built with TF-IDF · Cosine Similarity · FastAPI &nbsp;·&nbsp; Riochi511</footer>

<script>
  let activeTab = 'text';
  let selectedFile = null;

  function switchTab(tab) {
    activeTab = tab;
    document.getElementById('tab-text').style.display = tab === 'text' ? 'block' : 'none';
    document.getElementById('tab-file').style.display = tab === 'file' ? 'block' : 'none';
    document.querySelectorAll('.tab-btn').forEach((b, i) => {
      b.classList.toggle('active', (i === 0 && tab === 'text') || (i === 1 && tab === 'file'));
    });
  }

  function handleFile(input) {
    selectedFile = input.files[0];
    document.getElementById('file-name').textContent = selectedFile ? '✓ ' + selectedFile.name : '';
  }

  // Drag and drop
  const dropZone = document.getElementById('drop-zone');
  dropZone.addEventListener('dragover', e => { e.preventDefault(); dropZone.classList.add('dragover'); });
  dropZone.addEventListener('dragleave', () => dropZone.classList.remove('dragover'));
  dropZone.addEventListener('drop', e => {
    e.preventDefault();
    dropZone.classList.remove('dragover');
    const file = e.dataTransfer.files[0];
    if (file) {
      selectedFile = file;
      document.getElementById('file-name').textContent = '✓ ' + file.name;
    }
  });

  function scoreClass(score) {
    if (score >= 0.35) return 'score-high';
    if (score >= 0.15) return 'score-medium';
    return 'score-low';
  }

  async function submitResume() {
    const btn = document.getElementById('submit-btn');
    const errorBox = document.getElementById('error-box');
    errorBox.style.display = 'none';

    const topN = document.getElementById('top-n').value;
    const formData = new FormData();
    formData.append('top_n', topN);

    if (activeTab === 'text') {
      const text = document.getElementById('resume-text').value.trim();
      if (!text) { showError('Please paste your resume text before searching.'); return; }
      formData.append('resume_text', text);
    } else {
      if (!selectedFile) { showError('Please select a PDF or DOCX file.'); return; }
      formData.append('resume_file', selectedFile);
    }

    btn.disabled = true;
    btn.textContent = 'Analysing…';
    document.getElementById('loading').style.display = 'block';
    document.getElementById('results').style.display = 'none';

    try {
      const res = await fetch('/match', { method: 'POST', body: formData });
      const data = await res.json();
      if (!res.ok) { showError(data.detail || 'Something went wrong.'); return; }
      renderResults(data.matches);
    } catch (err) {
      showError('Could not reach the server. Please try again.');
    } finally {
      btn.disabled = false;
      btn.textContent = 'Find Matching Jobs';
      document.getElementById('loading').style.display = 'none';
    }
  }

  function renderResults(matches) {
    const list = document.getElementById('results-list');
    document.getElementById('results-count').textContent = matches.length + ' roles found';
    list.innerHTML = matches.map(m => `
      <div class="job-card">
        <div class="rank-num">#${m.rank}</div>
        <div class="job-info">
          <div class="job-title">${m.title}</div>
          <div class="job-company">${m.company}</div>
          <div class="job-preview">${m.preview}</div>
        </div>
        <div class="score-pill ${scoreClass(m.similarity)}">${(m.similarity * 100).toFixed(1)}% match</div>
      </div>
    `).join('');
    document.getElementById('results').style.display = 'block';
    document.getElementById('results').scrollIntoView({ behavior: 'smooth', block: 'start' });
  }

  function showError(msg) {
    const box = document.getElementById('error-box');
    box.textContent = msg;
    box.style.display = 'block';
    document.getElementById('loading').style.display = 'none';
  }
</script>
</body>
</html>
"""

# ── Text extraction helpers ───────────────────────────────────────────────────
def extract_text_from_pdf(file_bytes):
    text = ""
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text.strip()

def extract_text_from_docx(file_bytes):
    doc = docx.Document(io.BytesIO(file_bytes))
    return "\n".join(p.text for p in doc.paragraphs if p.text.strip())

def extract_resume_text(file):
    file_bytes = file.file.read()
    if file.filename.lower().endswith(".pdf"):
        return extract_text_from_pdf(file_bytes)
    elif file.filename.lower().endswith(".docx"):
        return extract_text_from_docx(file_bytes)
    else:
        raise HTTPException(status_code=400, detail="Upload a PDF or DOCX file.")

# ── Matching logic ────────────────────────────────────────────────────────────
def match_jobs(resume_text, top_n=10):
    resume_vec   = vectorizer.transform([resume_text])
    similarities = cosine_similarity(resume_vec, tfidf_matrix).flatten()
    top_indices  = np.argsort(similarities)[::-1][:top_n]
    results = []
    for rank, idx in enumerate(top_indices, start=1):
        row = jobs_df.iloc[idx]
        results.append({
            "rank":       rank,
            "title":      str(row.get("title", "N/A")),
            "company":    str(row.get("company_name", "N/A")),
            "preview":    str(row.get("description", ""))[:150] + "...",
            "similarity": round(float(similarities[idx]), 4),
        })
    return results

# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
def root():
    return HTML

@app.post("/match")
async def match(
    resume_text: Optional[str]        = Form(None),
    resume_file: Optional[UploadFile] = File(None),
    top_n:       int                  = Form(10),
):
    if resume_file and resume_file.filename:
        text = extract_resume_text(resume_file)
    elif resume_text and resume_text.strip():
        text = resume_text.strip()
    else:
        raise HTTPException(status_code=422, detail="Provide resume_text or upload a PDF/DOCX.")
    if not text:
        raise HTTPException(status_code=400, detail="Could not extract text from file.")
    matches = match_jobs(text, top_n=top_n)
    return JSONResponse(content={"matches": matches, "total": len(matches)})