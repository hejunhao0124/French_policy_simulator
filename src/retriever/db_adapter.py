"""
数据库适配器抽象接口
"""

from abc import ABC, abstractmethod
from typing import Optional
import pandas as pd


class DatabaseAdapter(ABC):
    """数据库适配器抽象接口"""

    @abstractmethod
    def save(self, query_id: str, question: str, results_df: pd.DataFrame) -> None:
        """保存查询结果到数据库"""
        ...

    @abstractmethod
    def get_cached_query(self, question: str) -> Optional[str]:
        """检查是否有相同问题的缓存，返回 query_id 或 None"""
        ...

    @abstractmethod
    def load(self, query_id: str) -> pd.DataFrame:
        """加载某次查询的结果"""
        ...

    @abstractmethod
    def list_queries(self) -> pd.DataFrame:
        """列出所有查询记录"""
        ...

    @abstractmethod
    def get_question(self, query_id: str) -> Optional[str]:
        """获取某次查询的原始问题"""
        ...

    @abstractmethod
    def delete_query(self, query_id: str) -> None:
        """删除某次查询及其回答记录"""
        ...

    @abstractmethod
    def close(self) -> None:
        """关闭数据库连接"""
        ...
