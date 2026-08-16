import os
import re
import csv
import json
import argparse
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")

LLM_PROVIDER = None
LLM_AVAILABLE = False

if OPENAI_API_KEY:
    import openai
    client = openai.OpenAI(api_key=OPENAI_API_KEY)
    LLM_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    LLM_PROVIDER = "openai"
    LLM_AVAILABLE = True
elif GROQ_API_KEY:
    import groq
    client = groq.Groq(api_key=GROQ_API_KEY)
    LLM_MODEL = os.environ.get("GROQ_MODEL", "llama3-70b-8192")
    LLM_PROVIDER = "groq"
    LLM_AVAILABLE = True
elif ANTHROPIC_API_KEY:
    import anthropic
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    LLM_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-6")
    LLM_PROVIDER = "anthropic"
    LLM_AVAILABLE = True


def read_resume_text(path: Path) -> str:
    suffix = path.suffix.lower()
    
    if suffix == ".txt":
        return path.read_text(encoding="utf-8", errors="ignore")
    
    if suffix == ".pdf":
        import pdfplumber
        text = []
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                if page_text := page.extract_text():
                    text.append(page_text)
        return "\n".join(text)
    
    if suffix == ".docx":
        import docx
        doc = docx.Document(path)
        return "\n".join(p.text for p in doc.paragraphs)
    
    raise ValueError(f"Unsupported file type: {suffix}")


def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def compute_tfidf_scores(jd_text: str, resume_texts: list[str]) -> list[float]:
    corpus = [jd_text] + resume_texts
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    tfidf_matrix = vectorizer.fit_transform(corpus)

    jd_vector = tfidf_matrix[0:1]
    resume_vectors = tfidf_matrix[1:]

    sims = cosine_similarity(jd_vector, resume_vectors)[0]
    return [round(float(s) * 100, 2) for s in sims]


LLM_SYSTEM_PROMPT = """You are a meticulous technical recruiter.
You will be given a JOB DESCRIPTION and one CANDIDATE RESUME.

Your job:
1. Extract the candidate's key skills, total years of relevant experience (estimate if not explicit), and highest education level.
2. Score the candidate's relevance to the job on a 0-100 scale, judging skill overlap, experience level, and domain fit.
3. List 2-3 concrete strengths and 2-3 concrete gaps relative to the JD.
4. Write a one-sentence overall verdict.

Respond ONLY with valid JSON in this shape:
{
  "skills": ["skill1", "skill2"],
  "years_experience": 2,
  "education": "B.Tech Computer Science",
  "llm_score": 78,
  "strengths": ["...", "..."],
  "gaps": ["...", "..."],
  "verdict": "..."
}"""


def score_with_llm(jd_text: str, resume_text: str) -> dict:
    if not LLM_AVAILABLE:
        return {
            "skills": [],
            "years_experience": None,
            "education": None,
            "llm_score": None,
            "strengths": [],
            "gaps": [],
            "verdict": "LLM scoring skipped (no API key set).",
        }

    user_msg = f"JOB DESCRIPTION:\n{jd_text}\n\nCANDIDATE RESUME:\n{resume_text}"

    if LLM_PROVIDER in ("openai", "groq"):
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": LLM_SYSTEM_PROMPT},
                {"role": "user", "content": user_msg}
            ],
            response_format={"type": "json_object"},
        )
        raw = response.choices[0].message.content.strip()
    else:
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


def combine_scores(tfidf_score: float, llm_score) -> float:
    if llm_score is None:
        return round(tfidf_score, 2)
    return round(0.4 * tfidf_score + 0.6 * llm_score, 2)


def run(jd_path: str, resumes_dir: str, out_dir: str):
    jd_path, resumes_dir, out_dir = Path(jd_path), Path(resumes_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    jd_text = clean_text(read_resume_text(jd_path))
    resume_files = sorted(
        [p for p in resumes_dir.iterdir() if p.suffix.lower() in (".txt", ".pdf", ".docx")]
    )
    
    if not resume_files:
        raise SystemExit(f"No resumes found in {resumes_dir}")

    print(f"Loaded JD from {jd_path}")
    print(f"Found {len(resume_files)} resumes to score")
    print(f"LLM scoring is {'ENABLED (' + LLM_MODEL + ')' if LLM_AVAILABLE else 'DISABLED (TF-IDF only)'}\n")

    resume_texts = [clean_text(read_resume_text(p)) for p in resume_files]
    tfidf_scores = compute_tfidf_scores(jd_text, resume_texts)

    results = []
    keys_to_extract = ["skills", "years_experience", "education", "strengths", "gaps", "verdict"]
    
    for path, text, tfidf_score in zip(resume_files, resume_texts, tfidf_scores):
        print(f"Scoring {path.name}...")
        llm_result = score_with_llm(jd_text, text)
        final_score = combine_scores(tfidf_score, llm_result.get("llm_score"))

        result_data = {
            "candidate": path.stem,
            "file": path.name,
            "final_score": final_score,
            "tfidf_score": tfidf_score,
            "llm_score": llm_result.get("llm_score")
        }
        
        for k in keys_to_extract:
            result_data[k] = llm_result.get(k, [] if k in ("skills", "strengths", "gaps") else None)
            
        results.append(result_data)

    results.sort(key=lambda r: r["final_score"], reverse=True)
    for i, r in enumerate(results, start=1):
        r["rank"] = i

    json_path = out_dir / "ranked_results.json"
    json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

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
                r["llm_score"], r.get("years_experience"), r.get("education"),
                "; ".join(r.get("skills", [])), r.get("verdict")
            ])

    print("\n--- Ranked Shortlist ---")
    for r in results:
        print(f"#{r['rank']:2d} {r['candidate']:<20} | Final: {r['final_score']:<6} (TF-IDF: {r['tfidf_score']:<6}, LLM: {r['llm_score']})")

    print(f"\nResults saved to {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Resume Screener")
    parser.add_argument("--jd", default="data/job_description.txt", help="Job description file path")
    parser.add_argument("--resumes", default="data/resumes", help="Folder containing resumes")
    parser.add_argument("--out", default="output", help="Output directory")
    args = parser.parse_args()

    run(args.jd, args.resumes, args.out)
