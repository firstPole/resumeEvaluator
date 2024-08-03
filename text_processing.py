import spacy
import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import BertTokenizer, BertModel
from docx import Document
import PyPDF2
from nltk.tokenize import word_tokenize
from collections import Counter
from nltk.corpus import stopwords
import openai
# Load the spaCy model for entity extraction
nlp = spacy.load("en_core_web_md")

# Load BERT for semantic analysis
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')

# Ensure NLTK stopwords are downloaded
import nltk
nltk.download('stopwords')

def extract_text_from_docx(file_stream):
    doc = Document(file_stream)
    return '\n'.join([p.text for p in doc.paragraphs])

def extract_text_from_pdf(file_stream):
    try:
        reader = PyPDF2.PdfReader(file_stream)
        text = ''
        for page in reader.pages:
            text += page.extract_text()
        return text
    except Exception as e:
        return None



def preprocess_text(text):
    return ' '.join(text.split())

def extract_skills(text):
    doc = nlp(text)
    skills = set()
    for ent in doc.ents:
        if ent.label_ in ['ORG', 'PRODUCT', 'SKILL']:
            skills.add(ent.text.strip())
    return skills

def get_bert_embeddings(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = bert_model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).numpy()

def calculate_matching_score(resume_text, job_description):
    vectorizer = TfidfVectorizer(stop_words='english')
    vectors = vectorizer.fit_transform([resume_text, job_description])
    tfidf_similarity = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
    
    resume_embeddings = get_bert_embeddings(resume_text)
    job_embeddings = get_bert_embeddings(job_description)
    semantic_similarity = cosine_similarity(resume_embeddings, job_embeddings)[0][0]
    
    final_score = (tfidf_similarity + semantic_similarity) / 2.0
    
    return final_score * 100


def filter_relevant_skills(matching_tags, non_matching_tags):
    return non_matching_tags

def get_keyword_density(text):
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text.lower())
    tokens = [token for token in tokens if token.isalnum() and token not in stop_words]
    frequency = Counter(tokens)
    return frequency.most_common(8)
def count_tag_occurrences(text, tags):
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text.lower())
    tokens = [token for token in tokens if token.isalnum() and token not in stop_words]
    token_counts = Counter(tokens)
    
    tag_occurrences = {}
    for tag in tags:
        tag_occurrences[tag] = token_counts.get(tag.lower(), 0)
    
    return tag_occurrences

def update_resume_with_tags_using_llm(resume_text, selected_tags):
    prompt = (
        f"Here is a resume:\n\n{resume_text}\n\n"
        f"Here are some skills to add:\n{', '.join(selected_tags)}\n\n"
        f"Please integrate these skills into the resume in appropriate sections. "
        f"Ensure that the resume remains ATS optimized and the content is clear and professional."
    )

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=1500,
        temperature=0.7
    )

    updated_resume_text = response.choices[0].message['content'].strip()
    return updated_resume_text

def get_skill_use_cases_using_llm(tags):
    prompt = (
        f"Here are some skills:\n{', '.join(tags)}\n\n"
        f"For each skill, provide a real-world use case or example in 2-3 lines."
    )

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a knowledgeable assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=1500,
        temperature=0.7
    )

    return response.choices[0].message['content'].strip()

def save_text_to_docx(text, path):
    doc = Document()
    paragraphs = text.split('\n')
    for paragraph in paragraphs:
        if paragraph.strip():
            doc.add_paragraph(paragraph)
    doc.save(path)

def generate_personalized_recommendations(resume_text, job_description, matched_skills, missing_skills):
    # Use LLM to generate personalized recommendations
    prompt = (
        f"Here is a resume:\n\n{resume_text}\n\n"
        f"Here is a job description:\n\n{job_description}\n\n"
        f"Matched skills: {', '.join(matched_skills)}\n"
        f"Missing skills: {', '.join(missing_skills)}\n\n"
        f"Provide personalized recommendations for the candidate to improve their resume and better match the job description."
    )

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are an expert career advisor."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=1500,
        temperature=0.7
    )

    recommendations = response.choices[0].message['content'].strip()
    return recommendations
