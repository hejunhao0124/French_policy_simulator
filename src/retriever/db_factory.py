"""
数据库适配器工厂
"""

from src.utils import Config
from .db_adapter import DatabaseAdapter
from .sqlite_adapter import SQLiteAdapter
from .sqlserver_adapter import SQLServerAdapter


def create_database_adapter() -> DatabaseAdapter:
    """
    工厂函数：根据配置创建数据库适配器

    返回:
        DatabaseAdapter: SQLite 或 SQL Server 适配器实例
    """
    db_type = Config.DB_TYPE
    if db_type == "sqlserver":
        return SQLServerAdapter()
    else:
        return SQLiteAdapter(Config.DB_PATH)
