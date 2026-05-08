import hashlib
import sqlite3

def calculate_pdf_hash(pdf_path):
    """Generates a unique SHA-256 hash for the PDF file."""
    sha256_hash = hashlib.sha256()
    with open(pdf_path, "rb") as f:
        # Read file in chunks to handle large PDFs
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def get_paper_name_from_id(db_path, paper_id):
    """Stateless helper to look up a title by hash."""
    conn = sqlite3.connect(db_path)
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT title FROM papers WHERE paper_id = ?", (paper_id,))
        result = cursor.fetchone()
        return str(result[0]).replace(".pdf", "") if result else "Unknown Study"
    finally:
        conn.close()