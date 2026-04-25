"""
SQLite 数据库适配器
"""

import os
import sqlite3
import pandas as pd

from .db_adapter import DatabaseAdapter


class SQLiteAdapter(DatabaseAdapter):
    """SQLite 数据库实现"""

    def __init__(self, db_path="data/results.db"):
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self._create_tables()

    def _create_tables(self):
        """创建数据表（如果不存在）"""
        cur = self.conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS queries (
                query_id TEXT PRIMARY KEY,
                question TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                total_responses INTEGER
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS responses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query_id TEXT,
                persona_id TEXT,
                llm_response TEXT,
                support_score REAL,
                stance TEXT,
                similarity REAL,
                age INTEGER,
                sex TEXT,
                occupation TEXT,
                education_level TEXT,
                departement TEXT,
                FOREIGN KEY (query_id) REFERENCES queries(query_id)
            )
        """)
        self.conn.commit()

    def save(self, query_id, question, results_df):
        cur = self.conn.cursor()
        cur.execute(
            "INSERT OR REPLACE INTO queries (query_id, question, total_responses) VALUES (?, ?, ?)",
            (query_id, question, len(results_df)),
        )
        columns = [
            "query_id", "persona_id", "llm_response", "support_score",
            "stance", "similarity", "age", "sex", "occupation", "education_level", "departement",
        ]
        rows = []
        for _, row in results_df.iterrows():
            rows.append((
                query_id,
                row.get("persona_id", ""),
                row.get("llm_response", ""),
                row.get("support_score", None),
                row.get("stance", ""),
                row.get("similarity", None),
                row.get("age", None),
                row.get("sex", ""),
                row.get("occupation", ""),
                row.get("education_level", ""),
                row.get("departement", ""),
            ))
        cur.executemany(
            "INSERT INTO responses (" + ",".join(columns) + ") VALUES (" + ",".join(["?"] * len(columns)) + ")",
            rows,
        )
        self.conn.commit()

    def get_cached_query(self, question):
        cur = self.conn.cursor()
        cur.execute("SELECT query_id FROM queries WHERE question = ?", (question,))
        row = cur.fetchone()
        return row[0] if row else None

    def load(self, query_id):
        return pd.read_sql_query(
            "SELECT * FROM responses WHERE query_id = ?",
            self.conn,
            params=(query_id,),
        )

    def list_queries(self):
        return pd.read_sql_query("SELECT * FROM queries ORDER BY timestamp DESC", self.conn)

    def get_question(self, query_id):
        cur = self.conn.cursor()
        cur.execute("SELECT question FROM queries WHERE query_id = ?", (query_id,))
        row = cur.fetchone()
        return row[0] if row else None

    def delete_query(self, query_id):
        cur = self.conn.cursor()
        cur.execute("DELETE FROM responses WHERE query_id = ?", (query_id,))
        cur.execute("DELETE FROM queries WHERE query_id = ?", (query_id,))
        self.conn.commit()

    def close(self):
        self.conn.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
