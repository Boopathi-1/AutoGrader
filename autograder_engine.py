# autograder_engine.py

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def grade_answer(model_answer, student_answer):
    """
    Grade the student's answer based on the cosine similarity between the model answer and the student's answer.
    Returns a score (0-10) and similarity score (0.0 - 1.0).
    """
    # Vectorize both answers using TF-IDF
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([model_answer, student_answer])
    
    # Calculate cosine similarity
    similarity_matrix = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
    
    # Convert similarity to a score
    similarity_score = similarity_matrix[0][0]
    
    # Basic scoring logic: the higher the similarity, the higher the score
    score = round(similarity_score * 10)  # Max score is 10
    
    return score, similarity_score
