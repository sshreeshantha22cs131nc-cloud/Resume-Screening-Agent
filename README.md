# Resume Screening Agent

Ranks a folder of resumes against a Job Description and outputs a scored,
ordered shortlist with reasoning for each candidate.

Built for the Rooman Technologies 24-Hour AI Agent Challenge — Category 1,
**Resume Screening Agent (Intermediate)**.

---

## What it does

1. Parses resumes in **PDF, DOCX, or TXT** format.
2. Scores each resume against the Job Description using a **hybrid method**:
   - **TF-IDF + cosine similarity** — a classic, local, explainable NLP
     similarity method (no API calls, near-instant even on low-spec
     hardware).
   - **LLM relevance scoring** — Claude reads the resume and JD, extracts
     skills / experience / education, and gives a 0–100 relevance score
     with a written verdict, strengths, and gaps.
3. Combines both into a **final weighted score** (40% TF-IDF, 60% LLM).
4. Outputs a ranked shortlist as both **JSON** (full detail) and **CSV**
   (spreadsheet-friendly summary).

---

## 1. Install

Requires Python 3.9+.

```bash
git clone <your-repo-url>
cd resume-screening-agent
pip install -r requirements.txt
```

## 2. Configure your API key

The agent uses the Anthropic (Claude) API for the semantic scoring step.

```bash
cp .env.example .env
# then open .env and paste your key:
# ANTHROPIC_API_KEY=sk-ant-xxxxxxxx
```

Get a key at https://console.anthropic.com/

> **No key? No problem.** The agent detects a missing key automatically and
> falls back to TF-IDF-only scoring, so it still runs end-to-end. This was
> a deliberate design choice — see Tradeoffs below.

## 3. Run it

```bash
python agent.py --jd data/job_description.txt --resumes data/resumes --out output
```

Flags (all optional, shown with defaults):

| Flag         | Default                | Description                          |
|--------------|-------------------------|---------------------------------------|
| `--jd`       | `data/job_description.txt` | Path to the job description file  |
| `--resumes`  | `data/resumes`          | Folder of resumes (.txt/.pdf/.docx)  |
| `--out`      | `output`                | Where to write results                |

## 4. Check the output

- `output/ranked_results.json` — full detail per candidate (scores, skills,
  strengths, gaps, verdict)
- `output/ranked_results.csv` — flat summary, easy to open in Excel/Sheets

Console output also prints a live ranked summary as it runs.

---

## Sample data included

- `data/job_description.txt` — a Data Analyst Intern JD (healthcare
  domain, modeled on a real internship posting)
- `data/resumes/` — 10 synthetic sample resumes with **deliberately varied
  fit levels**: strong healthcare-analytics matches, generalist data
  candidates, and clearly unrelated profiles (e.g. mechanical engineering,
  content writing) — to demonstrate the ranking actually discriminates
  between good and bad fits rather than scoring everything similarly.

Run the command above as-is to reproduce a full demo with no setup beyond
the API key.

---

## Scoring Method (why this approach)

**TF-IDF cosine similarity** (required "NLP similarity" component of the
rubric):
- Represents the JD and every resume as weighted word/phrase vectors
  (unigrams + bigrams) and measures the cosine angle between them.
- Fully local — no API cost, runs in milliseconds, works even with zero
  internet access. This matters for reproducibility during grading and for
  running comfortably on modest hardware.
- Weakness: it only sees surface-level word overlap. A resume that says
  "dashboarding in Looker" won't strictly match a JD asking for "Power BI"
  even though they're related skills — TF-IDF has no semantic understanding.

**LLM relevance scoring** (the "model choice" component):
- Claude reads the full resume and JD and judges fit the way a human
  recruiter would: does the *experience level* match, are the skills
  *actually relevant* even if worded differently, is there any domain
  fit (e.g. healthcare-adjacent projects for a healthcare JD)?
- Also extracts structured fields (skills list, years of experience,
  education) and produces human-readable strengths/gaps — this is what
  makes the final ranked list genuinely useful to a recruiter, not just a
  number.

**Why combine them (40% TF-IDF / 60% LLM)** rather than use just one:
- TF-IDF alone is fast and free but literal — it can be gamed by keyword-
  stuffing and misses genuine skill adjacency.
- LLM alone is smarter but non-deterministic — the same resume can get a
  slightly different score on different runs, and it's a black box with no
  local fallback.
- The blend keeps a stable, explainable numeric floor (TF-IDF) while
  letting the LLM's judgment do most of the differentiating work (60%
  weight), since it's the more accurate signal of true fit.

---

## Tradeoffs & What I'd Improve With More Time

- **Fallback design**: I made LLM scoring optional (auto-disables without
  an API key) rather than mandatory, so the agent is always runnable end-
  to-end by a reviewer even before they set up billing. Tradeoff: without
  the key, scores are TF-IDF-only and lose the semantic reasoning layer.
- **No resume "sectioning"**: I extract raw text rather than parsing into
  structured sections (Experience / Education / Skills) before scoring.
  With more time I'd add section detection so the LLM prompt could weight
  each section explicitly (e.g. penalize resumes that only list skills in
  a wordbank at the bottom with no supporting project evidence).
- **Single JD, batch resumes**: the agent currently scores many resumes
  against one JD in a single run. A natural extension would be batch mode
  across multiple JDs at once for teams screening several open roles.
- **No OCR for scanned/image-based PDFs**: `pdfplumber` extracts text from
  digitally generated PDFs but won't handle scanned image resumes. Adding
  an OCR fallback (e.g. `pytesseract`) would close this gap.
- **LLM cost/rate limits at scale**: with 10 resumes this is cheap and
  fast; at hundreds of resumes I'd batch requests or cache repeated JD
  embeddings, and consider a cheaper model for the extraction step,
  reserving the strongest model for final ranking of a shortlist.
- **Weighting is fixed (40/60)**: with more time I'd make the TF-IDF/LLM
  weight configurable per role, since some roles (e.g. highly technical
  keyword-driven roles) may want more weight on literal keyword match.

---

## Project Structure

```
resume-screening-agent/
├── agent.py                  # main pipeline
├── requirements.txt
├── .env.example
├── README.md
├── data/
│   ├── job_description.txt   # sample JD (GE HealthCare Data Analyst Intern)
│   └── resumes/               # 10 sample resumes, varied fit levels
└── output/
    ├── ranked_results.json   # generated on run
    └── ranked_results.csv    # generated on run
```
