"""
相似度检索模块
"""

import numpy as np
import pandas as pd


def search_similar(query, model, index, df, k=10000):
    """
    检索与问题最相似的 k 条人物

    参数:
        query: 用户问题（文本）
        model: Sentence-BERT 模型
        index: FAISS 索引
        df: 人物数据 DataFrame
        k: 返回数量

    返回:
        pd.DataFrame: 检索结果，包含原始数据 + similarity 字段
    """
    # 1. 对查询做 embedding
    query_vec = model.encode(
        [query],
        normalize_embeddings=True,
    ).astype(np.float32)

    # 2. 设置 nprobe（IVF 索引需要）
    if hasattr(index, 'nprobe'):
        index.nprobe = 16

    # 3. FAISS 搜索
    k = min(k, index.ntotal)
    scores, indices = index.search(query_vec, k)

    # 4. 从 DataFrame 中取回原始记录
    results = df.iloc[indices[0]].copy().reset_index(drop=True)
    results["similarity"] = scores[0]

    # 确保有 persona_id 列
    if "persona_id" not in results.columns and "uuid" in results.columns:
        results["persona_id"] = results["uuid"].astype(str)

    # 5. 按相似度排序
    results = results.sort_values("similarity", ascending=False).reset_index(drop=True)

    return results
