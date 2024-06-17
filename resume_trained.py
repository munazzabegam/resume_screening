import os
import pandas as pd
from pdfminer.high_level import extract_text
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load the spaCy model for English
nlp = spacy.load("en_core_web_sm")

def extract_text_from_pdf(pdf_path):
    try:
        return extract_text(pdf_path)
    except Exception as e:
        print(f"Error extracting text from {pdf_path}: {e}")
        return ""

def preprocess_text(text):
    doc = nlp(text.lower())
    return ' '.join([token.lemma_ for token in doc if token.is_alpha and not token.is_stop])

# Sample data for simplicity
data = {
    'resume': [
        "Experienced software engineer with expertise in Python, machine learning, and data analysis.",
        "Data analyst with strong skills in Excel and SQL, looking to transition to data science.",
        "Software developer with experience in Java and web development.",
        "Entry-level programmer with a background in Python and C++.",
        "Machine learning engineer with a PhD in Computer Science."
    ],
    'job_description': [
        "We are looking for a software engineer with experience in Python, machine learning, and data analysis.",
        "We need a data analyst with strong skills in Excel and SQL.",
        "Looking for a software developer with experience in Java and web development.",
        "Seeking an entry-level programmer with a background in Python and C++.",
        "Hiring a machine learning engineer with a PhD in Computer Science."
    ],
    'label': [1, 1, 1, 1, 1]  # 1 indicates a good fit, 0 indicates not a good fit
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Preprocess resumes and job descriptions
df['resume'] = df['resume'].apply(preprocess_text)
df['job_description'] = df['job_description'].apply(preprocess_text)

# Combine resumes and job descriptions
texts = df['resume'] + ' ' + df['job_description']

# Vectorize text
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(texts)

# Labels
y = df['label']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a RandomForestClassifier
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")

# Function to predict suitability of a resume for a job description
def predict_suitability(resume_text, job_description_text, model, vectorizer):
    # Preprocess text
    resume_text = preprocess_text(resume_text)
    job_description_text = preprocess_text(job_description_text)
    
    # Combine and vectorize text
    combined_text = resume_text + ' ' + job_description_text
    text_vector = vectorizer.transform([combined_text])
    
    # Predict using the trained model
    prediction = model.predict(text_vector)
    return prediction[0]

# Example usage
new_resume = "Skilled software engineer with a strong background in Python and machine learning."
new_job_description = "We are hiring a software engineer with experience in Python and machine learning."
is_suitable = predict_suitability(new_resume, new_job_description, model, vectorizer)
print("Suitable" if is_suitable else "Not Suitable")
