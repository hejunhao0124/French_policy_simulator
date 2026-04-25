"""
构建 FAISS 索引模块
"""

import os
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


import sys


def _safe_print(msg):
    """安全的 print，兼容 Streamlit 环境"""
    try:
        if not sys.stdout.closed:
            print(msg)
    except (ValueError, OSError):
        pass


def build_embeddings(texts, model_name="all-MiniLM-L6-v2", batch_size=256):
    """
    生成 Embedding 向量

    参数:
        texts: 文本列表
        model_name: Sentence-BERT 模型名称
        batch_size: 批处理大小

    返回:
        (np.ndarray, SentenceTransformer): 向量矩阵 (n, 384) 和模型实例
    """
    _safe_print(f"  加载模型: {model_name}")
    model = SentenceTransformer(model_name)

    dim = model.get_sentence_embedding_dimension()
    n = len(texts)
    embeddings = np.zeros((n, dim), dtype=np.float32)

    _safe_print(f"  生成向量: {n} 条文本，维度 {dim}")
    for start in tqdm(range(0, n, batch_size), desc="  Embedding", unit="batch"):
        end = min(start + batch_size, n)
        embeddings[start:end] = model.encode(
            texts[start:end],
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
        ).astype(np.float32)

    return embeddings, model


def build_faiss_index(embeddings):
    """
    构建 FAISS 索引（使用 IndexFlatIP，向量已归一化等价于余弦相似度）

    参数:
        embeddings: 向量矩阵 (n, d)

    返回:
        faiss.Index: FAISS 索引对象
    """
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # 内积相似度（归一化后 = 余弦相似度）
    index.add(embeddings)
    return index


def build_faiss_index_ivf(embeddings, n_list=256):
    """
    构建 IVFFlat 索引（适合大规模数据，内存更省）

    参数:
        embeddings: 向量矩阵 (n, d)
        n_list: IVF 聚类中心数

    返回:
        faiss.Index: FAISS 索引对象
    """
    dim = embeddings.shape[1]
    quantizer = faiss.IndexFlatIP(dim)
    index = faiss.IndexIVFFlat(quantizer, dim, n_list, faiss.METRIC_INNER_PRODUCT)

    # 训练聚类
    index.train(embeddings)
    index.add(embeddings)

    # 设置搜索 nprobe
    index.nprobe = 16
    return index


def save_index(index, path="data/index.faiss"):
    """
    保存 FAISS 索引到磁盘

    参数:
        index: FAISS 索引对象
        path: 保存路径
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    faiss.write_index(index, path)
    _safe_print(f"  索引已保存: {path}")


def load_index(path="data/index.faiss"):
    """
    从磁盘加载 FAISS 索引

    参数:
        path: 索引文件路径

    返回:
        faiss.Index: FAISS 索引对象
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"索引文件不存在: {path}")
    index = faiss.read_index(path)
    _safe_print(f"  索引已加载: {path} (向量数: {index.ntotal})")
    return index
