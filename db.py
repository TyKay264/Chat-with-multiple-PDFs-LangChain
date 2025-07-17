import sqlite3
from datetime import datetime

DB_PATH = "vector_metadata.db"

def create_tables():
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_name TEXT UNIQUE,
                uploaded_at TEXT
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_id INTEGER,
                chunk_index INTEGER,
                text TEXT,
                FOREIGN KEY (document_id) REFERENCES documents(id)
            )
        ''')
        conn.commit()

def insert_document(file_name):
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("INSERT OR IGNORE INTO documents (file_name, uploaded_at) VALUES (?, ?)", 
                       (file_name, datetime.now().isoformat()))
        conn.commit()
        cursor.execute("SELECT id FROM documents WHERE file_name = ?", (file_name,))
        return cursor.fetchone()[0]

def insert_chunks(document_id, chunks):
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        for i, chunk in enumerate(chunks):
            cursor.execute("INSERT INTO chunks (document_id, chunk_index, text) VALUES (?, ?, ?)", 
                           (document_id, i, chunk))
        conn.commit()

def get_chunks_by_file(file_name):
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT c.text FROM chunks c
            JOIN documents d ON c.document_id = d.id
            WHERE d.file_name = ?
        """, (file_name,))
        return [row[0] for row in cursor.fetchall()]

def get_all_documents():
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT file_name FROM documents")
        return [row[0] for row in cursor.fetchall()]
