"""
SQL Server 数据库适配器
"""

import os
import pandas as pd
import pyodbc

from .db_adapter import DatabaseAdapter
from src.utils import Config


class SQLServerAdapter(DatabaseAdapter):
    """SQL Server 数据库实现"""

    def __init__(self):
        if Config.SQL_USER and Config.SQL_PASSWORD:
            # SQL Server 认证
            self.conn_str = (
                f"DRIVER={{{Config.SQL_DRIVER}}};"
                f"SERVER={Config.SQL_SERVER};"
                f"DATABASE={Config.SQL_DATABASE};"
                f"UID={Config.SQL_USER};"
                f"PWD={Config.SQL_PASSWORD};"
                f"TrustServerCertificate=yes;"
                f"Encrypt=no;"
            )
        else:
            # Windows 认证
            self.conn_str = (
                f"DRIVER={{{Config.SQL_DRIVER}}};"
                f"SERVER={Config.SQL_SERVER};"
                f"DATABASE={Config.SQL_DATABASE};"
                f"Trusted_Connection=yes;"
                f"TrustServerCertificate=yes;"
                f"Encrypt=no;"
            )
        self.conn = pyodbc.connect(self.conn_str, autocommit=False)
        self._create_tables()

    def _create_tables(self):
        """创建数据表（如果不存在）"""
        cur = self.conn.cursor()

        # 检查表是否存在
        cur.execute("""
            IF NOT EXISTS (SELECT * FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_NAME = 'queries')
            BEGIN
                CREATE TABLE queries (
                    query_id NVARCHAR(100) PRIMARY KEY,
                    question NVARCHAR(MAX),
                    timestamp DATETIME DEFAULT GETDATE(),
                    total_responses INT
                )
            END
        """)
        cur.execute("""
            IF NOT EXISTS (SELECT * FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_NAME = 'responses')
            BEGIN
                CREATE TABLE responses (
                    id INT IDENTITY(1,1) PRIMARY KEY,
                    query_id NVARCHAR(100) FOREIGN KEY REFERENCES queries(query_id),
                    persona_id NVARCHAR(100),
                    llm_response NVARCHAR(MAX),
                    support_score FLOAT,
                    stance NVARCHAR(20),
                    similarity FLOAT,
                    age INT,
                    sex NVARCHAR(20),
                    occupation NVARCHAR(200),
                    education_level NVARCHAR(200),
                    departement NVARCHAR(200)
                )
            END
        """)
        # 为已存在的表添加缺失列
        cur.execute("""
            IF COL_LENGTH('responses', 'sex') IS NULL
            BEGIN
                ALTER TABLE responses ADD sex NVARCHAR(20) NULL
            END
        """)
        self.conn.commit()

    def save(self, query_id, question, results_df):
        cur = self.conn.cursor()
        # 使用 MERGE 实现 upsert
        cur.execute(
            """MERGE queries AS target
            USING (SELECT ? AS query_id, ? AS question, ? AS total_responses) AS source
            ON target.query_id = source.query_id
            WHEN MATCHED THEN UPDATE SET question = source.question, total_responses = source.total_responses
            WHEN NOT MATCHED THEN INSERT (query_id, question, total_responses) VALUES (source.query_id, source.question, source.total_responses);""",
            (query_id, question, len(results_df)),
        )

        columns = [
            "query_id", "persona_id", "llm_response", "support_score",
            "stance", "similarity", "age", "sex", "occupation", "education_level", "departement",
        ]
        rows = []
        for _, row in results_df.iterrows():
            score = row.get("support_score")
            sim = row.get("similarity")
            age_val = row.get("age")
            # 过滤掉没有成功回答的行
            if row.get("stance") is None or pd.isna(score):
                continue
            rows.append((
                query_id,
                str(row.get("persona_id", "")),
                str(row.get("llm_response", "")),
                float(score) if score is not None else None,
                str(row.get("stance", "")),
                float(sim) if sim is not None else None,
                int(age_val) if age_val is not None else None,
                str(row.get("sex", "")),
                str(row.get("occupation", "")),
                str(row.get("education_level", "")),
                str(row.get("departement", "")),
            ))
        if not rows:
            return  # 无成功回答，跳过保存
        placeholders = ",".join(["?"] * len(columns))
        cur.executemany(
            f"INSERT INTO responses ({','.join(columns)}) VALUES ({placeholders})",
            rows,
        )
        self.conn.commit()

    def get_cached_query(self, question):
        cur = self.conn.cursor()
        cur.execute("SELECT query_id FROM queries WHERE question = ?", (question,))
        row = cur.fetchone()
        return row[0] if row else None

    def load(self, query_id):
        return pd.read_sql(
            "SELECT * FROM responses WHERE query_id = ?",
            self.conn,
            params=(query_id,),
        )

    def list_queries(self):
        return pd.read_sql(
            "SELECT * FROM queries ORDER BY timestamp DESC",
            self.conn,
        )

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
