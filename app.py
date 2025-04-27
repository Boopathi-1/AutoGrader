import streamlit as st
import pandas as pd
import requests
import sentence_transformers
import numpy as np
import os  # Import os for environment variables
from db import create_db, insert_marks, get_all_marks

# Create database
create_db()

# Load Sentence-BERT model for answer similarity
model = sentence_transformers.SentenceTransformer('all-MiniLM-L6-v2')

# Page setup
st.set_page_config(page_title="AutoGrader AI", page_icon="📘", layout="wide")

# Custom CSS
st.markdown("""
    <style>
        .main { background-color: #f8f9fa; }
        .stButton>button {
            color: white;
            background-color: #007BFF;
            border-radius: 8px;
            padding: 0.5rem 1rem;
            font-weight: bold;
        }
        .stDataFrame { background-color: white; border-radius: 10px; }
        h1, h3, h4, h5 { color: #343a40; text-align: center; }
        .block-container { padding-top: 2rem; }
    </style>
""", unsafe_allow_html=True)

# --- LLaMA 3 API Setup ---
LLAMA_API_KEY = "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"# Fetch API key from Streamlit secrets
LLAMA_API_URL = "https://openrouter.ai/api/v1/chat/completions"
LLAMA_MODEL = "meta-llama/llama-3-8b-instruct"

if not LLAMA_API_KEY:
    st.error("❌ Missing API key for LLaMA. Please set it in the secrets.toml file.")
    st.stop()

# --- Model Answers ---
questions = [
    {"question": "What is Machine Learning?", "model_answer": "Machine Learning is a subset of AI where machines learn from data.", "max_score": 10},
    {"question": "Define supervised learning.", "model_answer": "Supervised learning is where a model learns from labeled data.", "max_score": 5},
    {"question": "What is overfitting?", "model_answer": "Overfitting is when a model learns the noise in training data, leading to poor performance on new data.", "max_score": 5},
    {"question": "Explain the concept of gradient descent.", "model_answer": "Gradient descent is an optimization algorithm used to minimize the cost function by adjusting model parameters.", "max_score": 5},
    {"question": "What is a neural network?", "model_answer": "A neural network is a machine learning model that mimics the human brain, consisting of layers of nodes that process input data.", "max_score": 5}
]

# --- User Login ---
# Dummy users for the login system
users_db = {
    "student1": "password123",
    "student2": "password456",
}

# --- Login Form ---
def login():
    st.title("📘 Student Login")
    st.subheader("Please enter your username and password.")
    
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    
    if st.button("Login"):
        if username in users_db and users_db[username] == password:
            st.session_state.username = username
            st.success(f"Welcome, {username}!")
            return True
        else:
            st.error("❌ Invalid username or password. Please try again.")
            return False
    return False

# --- Sentence-BERT Based Grading ---
def grade_answer_with_sentence_bert(model_answer, student_answer):
    # Get sentence embeddings for model and student answers
    model_answer_embedding = model.encode([model_answer])
    student_answer_embedding = model.encode([student_answer])

    # Calculate cosine similarity
    similarity = np.dot(model_answer_embedding, student_answer_embedding.T) / (np.linalg.norm(model_answer_embedding) * np.linalg.norm(student_answer_embedding))
    score = round(similarity[0][0] * 10, 2)  # Scale it to a score out of 10
    return score, similarity[0][0]

# --- Get LLaMA Feedback ---
def get_llama_feedback(student_name, model_answer, student_answer, score):
    headers = {
        "Authorization": f"Bearer {LLAMA_API_KEY}",
        "Content-Type": "application/json"
    }
    feedback_prompt = (
        f"You are an AI assistant grading a student's answer.\n\n"
        f"Model Answer:\n{model_answer}\n\n"
        f"Student Answer:\n{student_answer}\n\n"
        f"Provide a short, friendly feedback for {student_name} based on the score. "
        f"If the score is less than 7, provide suggestions for improvement, "
        f"and if the score is 7 or more, give a positive comment and tips to keep improving.\n\n"
        f"Student Score: {score}"
    )
    
    payload = {
        "model": LLAMA_MODEL,
        "messages": [
            {"role": "user", "content": feedback_prompt}
        ],
        "max_tokens": 150,  # Short feedback
    }
    
    try:
        response = requests.post(LLAMA_API_URL, headers=headers, json=payload)
        response.raise_for_status()
        result = response.json()
        if "choices" in result and len(result["choices"]) > 0:
            feedback = result["choices"][0]["message"]["content"].strip()
            return feedback
        else:
            return f"⚠️ Unexpected API response: {result}"
    except Exception as e:
        return f"⚠️ LLaMA API error: {e}"

# --- Main Page ---
def main():
    st.markdown("""
        <h1>📘 AutoGrader AI - Powered by LLaMA 3 🦙</h1>
        <h5>Batch grading with instant, intelligent feedback for each student!</h5>
        <hr>
    """, unsafe_allow_html=True)
    
    if "username" not in st.session_state:
        # If no user is logged in, show the login page
        if not login():
            return  # Stop the function execution if login fails

    # Once logged in, let the student upload their answers
    st.markdown("### Answer Upload")
    uploaded_file = st.file_uploader("📤 Upload your answers in CSV format", type="csv")
    
    if uploaded_file:
        # Process the uploaded file
        df = pd.read_csv(uploaded_file)

        # Validate uploaded CSV
        required_columns = ['Name'] + [f"Q{i+1}" for i in range(len(questions))]
        if not all(col in df.columns for col in required_columns):
            st.error("❌ Uploaded CSV is missing required columns. Expected 'Name', 'Q1', 'Q2', etc.")
            st.stop()

        output_data = []

        progress = st.progress(0)
        for index, row in df.iterrows():
            student_name = row['Name']
            total_score = 0
            row_result = {"Name": student_name}

            # Collecting individual question feedback
            for i, q in enumerate(questions):
                q_key = f"Q{i+1}"
                if q_key not in row:
                    continue
                student_answer = row[q_key]
                if pd.isna(student_answer) or not isinstance(student_answer, str):
                    row_result[f"{q_key}_Score"] = 0
                    row_result[f"{q_key}_Similarity"] = 0.0
                    row_result[f"{q_key}_Feedback"] = "No answer provided."
                    continue

                score, similarity = grade_answer_with_sentence_bert(q['model_answer'], student_answer)
                feedback = get_llama_feedback(student_name, q['model_answer'], student_answer, score)

                row_result[f"{q_key}_Score"] = score
                row_result[f"{q_key}_Similarity"] = round(similarity, 2)
                row_result[f"{q_key}_Feedback"] = feedback
                total_score += score

            row_result["Total_Score"] = total_score
            output_data.append(row_result)

            # Insert total score into the database
            insert_marks(student_name, total_score)

            # Update progress
            progress.progress((index + 1) / len(df))

        output_df = pd.DataFrame(output_data)
        
        # Display Summary and Individual Feedback
        st.success("✅ Grading complete!")
        st.markdown("### 📊 Results Overview")
        
        for idx, row in output_df.iterrows():
            student_name = row['Name']
            total_score = row["Total_Score"]
            
            st.markdown(f"#### {student_name} - Total Score: {round(total_score, 2)}/50")
            st.markdown("### Detailed Question Feedback:")
            for i, q in enumerate(questions):
                q_key = f"Q{i+1}"
                question_feedback = row[f"{q_key}_Feedback"]
                st.markdown(f"**Q{i+1}: {q['question']}**")
                st.markdown(f"{question_feedback}")
            
            st.markdown("---")

        # Download CSV Results
        csv = output_df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Results CSV", data=csv, file_name="graded_results.csv", mime="text/csv")

        # Display database records (optional)
        st.markdown("DataBase is still in Process")
        st.markdown("### Student Marks Stored in Database:")
        db_marks = get_all_marks()
        db_df = pd.DataFrame(db_marks, columns=["ID", "Name", "Total_Score"])
        st.write(db_df)


if __name__ == "__main__":
    main()
