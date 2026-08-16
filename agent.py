"""
Resume Screening Agent
-----------------------
Ranks a folder of resumes against a Job Description using a hybrid scoring
method: classic TF-IDF cosine similarity (fast, local, explainable) combined
with an LLM relevance judgement (semantic understanding + reasoning).

Usage:
    python agent.py --jd data/job_description.txt --resumes data/resumes --out output

Requires ANTHROPIC_API_KEY in your environment (or a .env file) for the LLM
scoring step. If no key is found, the agent automatically falls back to
TF-IDF-only scoring so it still runs end-to-end (see README "Tradeoffs").
"""

import os
import re
import csv
import json
import argparse
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------- optional .env support ----------
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ---------- optional LLM client ----------
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
LLM_AVAILABLE = bool(ANTHROPIC_API_KEY)

if LLM_AVAILABLE:
    import anthropic
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    LLM_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-6")


# =========================================================
# 1. FILE PARSING (PDF / DOCX / TXT)
# =========================================================
def read_resume_text(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".txt":
        return path.read_text(encoding="utf-8", errors="ignore")

    if suffix == ".pdf":
        import pdfplumber
        text = []
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text.append(page_text)
        return "\n".join(text)

    if suffix == ".docx":
        import docx
        d = docx.Document(path)
        return "\n".join(p.text for p in d.paragraphs)

    raise ValueError(f"Unsupported file type: {suffix}")


def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# =========================================================
# 2. TF-IDF COSINE SIMILARITY (the "NLP similarity method")
# =========================================================
def compute_tfidf_scores(jd_text: str, resume_texts: list[str]) -> list[float]:
    """
    Classic bag-of-words NLP similarity. Runs 100% locally, no API needed,
    negligible CPU/RAM cost -- fine on low-spec machines.
    Returns a 0-100 score per resume.
    """
    corpus = [jd_text] + resume_texts
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    tfidf_matrix = vectorizer.fit_transform(corpus)

    jd_vector = tfidf_matrix[0:1]
    resume_vectors = tfidf_matrix[1:]

    sims = cosine_similarity(jd_vector, resume_vectors)[0]
    return [round(float(s) * 100, 2) for s in sims]


# =========================================================
# 3. LLM SCORING (semantic relevance + structured extraction)
# =========================================================
LLM_SYSTEM_PROMPT = """You are a meticulous technical recruiter.
You will be given a JOB DESCRIPTION and one CANDIDATE RESUME.

Your job:
1. Extract the candidate's key skills, total years of relevant experience
   (estimate if not explicit), and highest education level.
2. Score the candidate's relevance to the job on a 0-100 scale, judging
   skill overlap, experience level, and domain fit -- not just keyword
   matches.
3. List 2-3 concrete strengths and 2-3 concrete gaps relative to the JD.
4. Write a one-sentence overall verdict.

Respond ONLY with valid JSON, no markdown fences, no preamble, in exactly
this shape:

{
  "skills": ["skill1", "skill2"],
  "years_experience": 2,
  "education": "B.Tech Computer Science",
  "llm_score": 78,
  "strengths": ["...", "..."],
  "gaps": ["...", "..."],
  "verdict": "..."
}
"""


def score_with_llm(jd_text: str, resume_text: str) -> dict:
    if not LLM_AVAILABLE:
        return {
            "skills": [],
            "years_experience": None,
            "education": None,
            "llm_score": None,
            "strengths": [],
            "gaps": [],
            "verdict": "LLM scoring skipped (no ANTHROPIC_API_KEY set).",
        }

    user_msg = f"JOB DESCRIPTION:\n{jd_text}\n\nCANDIDATE RESUME:\n{resume_text}"

    response = client.messages.create(
        model=LLM_MODEL,
        max_tokens=600,
        system=LLM_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_msg}],
    )

    raw = response.content[0].text.strip()
    raw = re.sub(r"^```json|```$", "", raw, flags=re.MULTILINE).strip()

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {
            "skills": [],
            "years_experience": None,
            "education": None,
            "llm_score": None,
            "strengths": [],
            "gaps": [],
            "verdict": f"Could not parse LLM output: {raw[:200]}",
        }


# =========================================================
# 4. COMBINE + RANK
# =========================================================
def combine_scores(tfidf_score: float, llm_score) -> float:
    """
    Weighted hybrid final score.
    - If LLM score is available: 40% TF-IDF (keyword/phrase grounding)
      + 60% LLM (semantic understanding of skills/experience fit).
    - If LLM unavailable: TF-IDF score alone.
    """
    if llm_score is None:
        return round(tfidf_score, 2)
    return round(0.4 * tfidf_score + 0.6 * llm_score, 2)


# =========================================================
# 5. MAIN PIPELINE
# =========================================================
def run(jd_path: str, resumes_dir: str, out_dir: str):
    jd_path = Path(jd_path)
    resumes_dir = Path(resumes_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    jd_text = clean_text(read_resume_text(jd_path))

    resume_files = sorted(
        [p for p in resumes_dir.iterdir() if p.suffix.lower() in (".txt", ".pdf", ".docx")]
    )
    if not resume_files:
        raise SystemExit(f"No resumes found in {resumes_dir}")

    print(f"Loaded JD from {jd_path}")
    print(f"Found {len(resume_files)} resumes in {resumes_dir}")
    print(f"LLM scoring: {'ENABLED (' + LLM_MODEL + ')' if LLM_AVAILABLE else 'DISABLED (no API key -- TF-IDF only)'}\n")

    resume_texts = [clean_text(read_resume_text(p)) for p in resume_files]
    tfidf_scores = compute_tfidf_scores(jd_text, resume_texts)

    results = []
    for path, text, tfidf_score in zip(resume_files, resume_texts, tfidf_scores):
        print(f"Scoring {path.name} ...")
        llm_result = score_with_llm(jd_text, text)
        final_score = combine_scores(tfidf_score, llm_result.get("llm_score"))

        results.append({
            "candidate": path.stem,
            "file": path.name,
            "final_score": final_score,
            "tfidf_score": tfidf_score,
            "llm_score": llm_result.get("llm_score"),
            "skills": llm_result.get("skills", []),
            "years_experience": llm_result.get("years_experience"),
            "education": llm_result.get("education"),
            "strengths": llm_result.get("strengths", []),
            "gaps": llm_result.get("gaps", []),
            "verdict": llm_result.get("verdict", ""),
        })

    results.sort(key=lambda r: r["final_score"], reverse=True)
    for i, r in enumerate(results, start=1):
        r["rank"] = i

    # ---- write JSON ----
    json_path = out_dir / "ranked_results.json"
    json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    # ---- write CSV ----
    csv_path = out_dir / "ranked_results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "rank", "candidate", "final_score", "tfidf_score", "llm_score",
            "years_experience", "education", "skills", "verdict"
        ])
        for r in results:
            writer.writerow([
                r["rank"], r["candidate"], r["final_score"], r["tfidf_score"],
                r["llm_score"], r["years_experience"], r["education"],
                "; ".join(r["skills"]), r["verdict"]
            ])

    # ---- console summary ----
    print("\n=== RANKED SHORTLIST ===")
    for r in results:
        print(f"#{r['rank']:2d}  {r['candidate']:<20}  final={r['final_score']:<6}  "
              f"tfidf={r['tfidf_score']:<6}  llm={r['llm_score']}")

    print(f"\nSaved: {json_path}")
    print(f"Saved: {csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Resume Screening Agent")
    parser.add_argument("--jd", default="data/job_description.txt", help="Path to job description file")
    parser.add_argument("--resumes", default="data/resumes", help="Folder of resumes (.txt/.pdf/.docx)")
    parser.add_argument("--out", default="output", help="Output folder for results")
    args = parser.parse_args()

    run(args.jd, args.resumes, args.out)
