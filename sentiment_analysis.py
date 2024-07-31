from textblob import TextBlob
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter
import re


def analyze_job_description(text):
    blob = TextBlob(text)
    language_tone = "Casual" if blob.sentiment.polarity < 0 else "Formal"
    
    if "team" in text.lower() or "collaborate" in text.lower():
        emphasis_teamwork = "Teamwork"
    else:
        emphasis_teamwork = "Individual Contribution"
    
    social_responsibility_terms = ["sustainability", "diversity", "ethics"]
    social_responsibility = any(term in text.lower() for term in social_responsibility_terms)
    
    keyword_density = get_keyword_density(text)
    
    return {
        "language_tone": language_tone,
        "emphasis_teamwork": emphasis_teamwork,
        "social_responsibility": social_responsibility,
        "keyword_density": keyword_density
    }

def get_keyword_density(text):
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text.lower())
    tokens = [token for token in tokens if token.isalnum() and token not in stop_words]
    frequency = Counter(tokens)
    return frequency.most_common(8)

def perform_sentiment_analysis(text):
    from textblob import TextBlob
    blob = TextBlob(text)
    sentiment = blob.sentiment.polarity
    if sentiment > 0:
        analysis = 'Positive'
    elif sentiment < 0:
        analysis = 'Negative'
    else:
        analysis = 'Neutral'
    
    return {'analysis': analysis}