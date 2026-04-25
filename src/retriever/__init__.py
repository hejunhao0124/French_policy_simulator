"""
检索模块
"""

from .db_adapter import DatabaseAdapter
from .sqlite_adapter import SQLiteAdapter
from .sqlserver_adapter import SQLServerAdapter
from .db_factory import create_database_adapter
from .build_index import build_embeddings, build_faiss_index, build_faiss_index_ivf, save_index, load_index
from .search import search_similar
from .save_results import ResultDatabase
