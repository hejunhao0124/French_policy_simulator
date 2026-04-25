"""
结果数据库模块
"""

import os
import sqlite3
import pandas as pd


class ResultDatabase:
    """结果数据库（SQLite）"""

    def __init__(self, db_path="data/results.db"):
        """
        初始化数据库连接

        参数:
            db_path: 数据库文件路径
        """
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self._create_tables()

    def _create_tables(self):
        """创建数据表（如果不存在）"""
        cur = self.conn.cursor()
        # 查询记录表
        cur.execute("""
            CREATE TABLE IF NOT EXISTS queries (
                query_id TEXT PRIMARY KEY,
                question TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                total_responses INTEGER
            )
        """)
        # 回答记录表
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
        """
        保存查询结果

        参数:
            query_id: 本次查询的唯一标识
            question: 用户问题
            results_df: 包含人物画像和 LLM 回答的 DataFrame
        """
        cur = self.conn.cursor()

        # 1. 保存查询记录
        cur.execute(
            "INSERT OR REPLACE INTO queries (query_id, question, total_responses) VALUES (?, ?, ?)",
            (query_id, question, len(results_df)),
        )

        # 2. 保存回答记录
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
        """
        检查是否有相同问题的缓存

        参数:
            question: 用户问题

        返回:
            query_id 或 None
        """
        cur = self.conn.cursor()
        cur.execute("SELECT query_id FROM queries WHERE question = ?", (question,))
        row = cur.fetchone()
        return row[0] if row else None

    def load(self, query_id):
        """
        加载某次查询的结果

        参数:
            query_id: 查询标识

        返回:
            pd.DataFrame: 该次查询的所有结果
        """
        return pd.read_sql_query(
            "SELECT * FROM responses WHERE query_id = ?",
            self.conn,
            params=(query_id,),
        )

    def list_queries(self):
        """
        列出所有查询记录

        返回:
            pd.DataFrame: 查询记录列表
        """
        return pd.read_sql_query("SELECT * FROM queries ORDER BY timestamp DESC", self.conn)

    def close(self):
        """关闭数据库连接"""
        self.conn.close()

    def __del__(self):
        """析构函数：确保关闭连接"""
        try:
            self.close()
        except Exception:
            pass
