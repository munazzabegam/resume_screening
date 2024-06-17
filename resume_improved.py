import os
from pdfminer.high_level import extract_text
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the spaCy model for English
nlp = spacy.load("en_core_web_sm")

def extract_text_from_pdf(pdf_path):
    try:
        return extract_text(pdf_path)
    except Exception as e:
        print(f"Error extracting text from {pdf_path}: {e}")
        return ""

# Directory containing resumes
resume_dir = "resumes"
resumes = []
for filename in os.listdir(resume_dir):
    if filename.endswith(".pdf"):
        file_path = os.path.join(resume_dir, filename)
        resumes.append((filename, extract_text_from_pdf(file_path)))

def extract_keywords(text):
    doc = nlp(text)
    keywords = [token.lemma_.lower() for token in doc if token.is_alpha and not token.is_stop]
    return keywords

def extract_entities(text):
    doc = nlp(text)
    entities = [ent.text.lower() for ent in doc.ents]
    return entities

# Synonyms dictionary
synonyms = {
    "python": ["python"],
    "machine learning": ["machine learning", "ml"],
    "data analysis": ["data analysis", "data analytics"],
    "problem-solving": ["problem solving", "troubleshooting"],
    "algorithms": ["algorithms", "algorithm"],
}

def expand_keywords(keywords):
    expanded = set()
    for keyword in keywords:
        if keyword in synonyms:
            expanded.update(synonyms[keyword])
        else:
            expanded.add(keyword)
    return expanded

# Job description text
job_description = """
We are looking for a software engineer with experience in Python, machine learning, and data analysis. 
The candidate should have good problem-solving skills and knowledge of algorithms.
"""

# Extract and expand job description keywords
job_keywords = expand_keywords(extract_keywords(job_description) + extract_entities(job_description))
print(f"Job Keywords: {job_keywords}")

# Extract and expand resume keywords
resume_keywords = []
for filename, text in resumes:
    keywords = expand_keywords(extract_keywords(text) + extract_entities(text))
    resume_keywords.append((filename, keywords))

# Combine job description and resume texts
all_texts = [job_description] + [text for _, text in resumes]

# Calculate TF-IDF
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(all_texts)

# Extract job description TF-IDF vector
job_tfidf = tfidf_matrix[0]

# Calculate resume scores based on TF-IDF
resume_scores = []
for i, (filename, _) in enumerate(resumes):
    score = (tfidf_matrix[i + 1] * job_tfidf.T).toarray()[0, 0]
    resume_scores.append((filename, score))

# Sort resumes by score
ranked_resumes = sorted(resume_scores, key=lambda x: x[1], reverse=True)

# Print ranked resumes
print("Ranked Resumes (TF-IDF):")
for rank, (filename, score) in enumerate(ranked_resumes, start=1):
    print(f"Rank {rank}: {filename} with score {score}")
