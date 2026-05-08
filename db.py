import sqlite3

import logging
logging.basicConfig(level=logging.INFO)

def init_db(db_path="sepsis_atlas.db"):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # 1. Paper Metadata
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS papers (
            paper_id TEXT PRIMARY KEY,
            title TEXT,
            doi TEXT,
            processed_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # 2. Figure Metadata (Visual Evidence Locker)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS figure_metadata (
            fig_id INTEGER PRIMARY KEY AUTOINCREMENT,
            paper_id TEXT,
            figure_name TEXT,
            caption TEXT,
            image_path TEXT,
            page_number INTEGER,
            FOREIGN KEY(paper_id) REFERENCES papers(paper_id)
        )
    ''')

    # 3. Structured Sepsis Data (The Spreadsheet)
    # Added figure_id to link every row to a specific image
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS observations (
            obs_id INTEGER PRIMARY KEY AUTOINCREMENT,
            paper_id TEXT,
            figure_id INTEGER, 
            study TEXT,
            population TEXT,
            sample_size TEXT,
            predictor TEXT,
            outcome TEXT,
            measurement_timing TEXT,
            method TEXT,
            effect_size TEXT,
            performance TEXT,
            notes TEXT,
            page_number INTEGER,
            FOREIGN KEY(paper_id) REFERENCES papers(paper_id),
            FOREIGN KEY(figure_id) REFERENCES figure_metadata(fig_id)
        )
    ''')

    # 4. Narrative Text (For RAG)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS text_chunks (
            chunk_id INTEGER PRIMARY KEY AUTOINCREMENT,
            paper_id TEXT,
            page_number INTEGER,
            chunk_content TEXT,
            FOREIGN KEY(paper_id) REFERENCES papers(paper_id)
        )
    ''')

    conn.commit()
    return conn

def save_paper_metadata(conn, paper_id, title):
    cursor = conn.cursor()
    cursor.execute("INSERT OR IGNORE INTO papers (paper_id, title) VALUES (?, ?)", (paper_id, title))
    conn.commit()

def save_figure_metadata(conn, paper_id, fig_name, caption, img_path, page_num):
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO figure_metadata (paper_id, figure_name, caption, image_path, page_number)
        VALUES (?, ?, ?, ?, ?)
    ''', (paper_id, fig_name, caption, str(img_path), page_num))
    conn.commit()
    return cursor.lastrowid # Returns the ID so we can link observations to it

def save_text_chunk(conn, paper_id, page_number, content):
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO text_chunks (paper_id, page_number, chunk_content) VALUES (?, ?, ?)",
        (paper_id, page_number, content)
    )
    conn.commit()

def save_observation(conn, paper_id, fig_id, page_num, obs):
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO observations (
            paper_id, figure_id, study, population, sample_size, predictor, outcome, 
            measurement_timing, method, effect_size, performance, notes, page_number
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        paper_id, fig_id, obs.study, obs.population, obs.sample_size, obs.predictor, 
        obs.outcome, obs.measurement_timing, obs.method, obs.effect_size, 
        obs.performance, obs.notes, page_num
    ))
    conn.commit()
    logging.info(f"💾 Saved observation for {obs.study} (Page {page_num})")