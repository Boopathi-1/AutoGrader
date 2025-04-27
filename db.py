import sqlite3

# Create a connection to the database
def create_db():
    conn = sqlite3.connect('student_marks.db')
    cursor = conn.cursor()
    
    # Create a table to store student marks (only name and total score)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS student_marks (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        total_score REAL NOT NULL
    )
    """)
    conn.commit()
    conn.close()

# Insert total score into the database
def insert_marks(name, total_score):
    conn = sqlite3.connect('student_marks.db')
    cursor = conn.cursor()
    
    # Insert student name and total score (no individual question scores)
    cursor.execute("""
    INSERT INTO student_marks (name, total_score)
    VALUES (?, ?)
    """, (name, total_score))
    conn.commit()
    conn.close()

# Get all marks from the database
def get_all_marks():
    conn = sqlite3.connect('student_marks.db')
    cursor = conn.cursor()
    
    cursor.execute("SELECT * FROM student_marks")
    rows = cursor.fetchall()
    
    conn.close()
    return rows
