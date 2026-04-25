"""
数据加载模块
"""

import os
os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')

import pandas as pd
from datasets import load_dataset
from datasets.utils.logging import disable_progress_bar
import warnings
warnings.filterwarnings('ignore')

from src.utils import Config


def load_full_data(sample_size=None, seed=42):
    """
    加载全量数据（通过 hf-mirror.com 镜像，流式模式）

    参数:
        sample_size: 采样数量，None=全量
        seed: 随机种子

    返回:
        pd.DataFrame: 原始数据
    """
    disable_progress_bar()
    print(f"正在从 HuggingFace (hf-mirror) 流式加载数据集...")
    dataset = load_dataset(Config.DATASET_NAME, split="train", streaming=True)

    if sample_size:
        # 使用 reservoir sampling 随机采样
        import random
        random.seed(seed)
        reservoir = []
        for i, row in enumerate(dataset):
            if len(reservoir) < sample_size:
                reservoir.append(row)
            else:
                j = random.randint(0, i)
                if j < sample_size:
                    reservoir[j] = row
            if (i + 1) % 100000 == 0:
                print(f"  已扫描 {i+1:,} 行...")
        print(f"  采样完成：{len(reservoir):,} 行")
        df = pd.DataFrame(reservoir)
    else:
        rows = list(dataset)
        df = pd.DataFrame(rows)
        print(f"  加载完成：{len(df):,} 行")

    return df


def load_local_data(path=None):
    """
    从本地文件加载数据（支持 parquet 和 csv）

    参数:
        path: 文件路径（默认使用 Config.DATA_PATH）

    返回:
        pd.DataFrame: 原始数据
    """
    path = path or Config.DATA_PATH
    if not os.path.exists(path):
        raise FileNotFoundError(f"本地数据文件不存在: {path}\n请在 .env 中设置 DATA_PATH=你的实际路径")
    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    elif path.endswith(".csv"):
        return pd.read_csv(path)
    else:
        raise ValueError(f"不支持的文件格式: {path}，请使用 .parquet 或 .csv")


def load_data_streaming(chunk_size=100000):
    """
    流式加载数据

    参数:
        chunk_size: 每批大小

    返回:
        generator: 每批数据的生成器
    """
    disable_progress_bar()
    dataset = load_dataset(Config.DATASET_NAME, split="train", streaming=True)
    batch = []
    for row in dataset:
        batch.append(row)
        if len(batch) >= chunk_size:
            yield pd.DataFrame(batch)
            batch = []
    if batch:
        yield pd.DataFrame(batch)