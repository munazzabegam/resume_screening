import os
from pdfminer.high_level import extract_text
import spacy

# Step 1: Setup
nlp = spacy.load("en_core_web_sm")

# Step 2: Data Collection
resume_dir = "resumes"
resumes = []
for filename in os.listdir(resume_dir):
    if filename.endswith(".pdf"):
        file_path = os.path.join(resume_dir, filename)
        resumes.append((filename, extract_text(file_path)))

# Step 3: Text Processing
def extract_keywords(text):
    doc = nlp(text)
    keywords = [token.text.lower() for token in doc if token.is_alpha and not token.is_stop]
    return keywords

job_description = """
We are looking for a software engineer with experience in Python, machine learning, and data analysis. 
The candidate should have good problem-solving skills and knowledge of algorithms.
"""
job_keywords = extract_keywords(job_description)

resume_keywords = []
for filename, text in resumes:
    keywords = extract_keywords(text)
    resume_keywords.append((filename, keywords))

# Step 4: Scoring
def calculate_score(job_keywords, resume_keywords):
    job_keywords_set = set(job_keywords)
    scores = []
    for filename, keywords in resume_keywords:
        resume_keywords_set = set(keywords)
        common_keywords = job_keywords_set & resume_keywords_set
        score = len(common_keywords)
        scores.append((filename, score))
    return scores

scores = calculate_score(job_keywords, resume_keywords)
ranked_resumes = sorted(scores, key=lambda x: x[1], reverse=True)

# Step 5: Output
print("Ranked Resumes:")
for rank, (filename, score) in enumerate(ranked_resumes, start=1):
    print(f"Rank {rank}: {filename} with score {score}")
