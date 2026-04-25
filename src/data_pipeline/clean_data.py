"""
数据清洗模块
"""

import os
import numpy as np
import pandas as pd


# 22 个字段定义
FIELDS_DEMOGRAPHIC = [
    "uuid", "sex", "age", "marital_status", "household_type",
    "education_level", "occupation", "commune", "departement", "country",
]

FIELDS_PERSONA = [
    "persona", "cultural_background", "professional_persona",
    "sports_persona", "arts_persona", "travel_persona", "culinary_persona",
]

FIELDS_EXTRAS = [
    "skills_and_expertise", "skills_and_expertise_list",
    "hobbies_and_interests", "hobbies_and_interests_list",
    "career_goals_and_ambitions",
]

ALL_FIELDS = FIELDS_DEMOGRAPHIC + FIELDS_PERSONA + FIELDS_EXTRAS


def clean_data(df):
    """
    清洗数据，保留所有字段，构造 LLM 和检索所需字段
    （优化版：使用向量化操作，避免 agg(axis=1)）

    参数:
        df: 原始 DataFrame

    返回:
        pd.DataFrame: 清洗后的数据
    """
    # 1. 保留已有字段，缺失的补空
    for col in ALL_FIELDS:
        if col not in df.columns:
            df = df.assign(**{col: None})

    # 2. 去重（基于 uuid）
    if "uuid" in df.columns:
        before = len(df)
        df = df.drop_duplicates(subset=["uuid"], keep="first")
        dupes = before - len(df)
        if dupes:
            print(f"  去重: 移除 {dupes:,} 条重复")

    # 3. 类型清洗 - age
    if "age" in df.columns:
        df["age"] = pd.to_numeric(df["age"], errors="coerce").fillna(0).clip(0, 120).astype(int)

    # 4. 字符串字段处理 - 向量化填充缺失值
    str_fields = [c for c in ALL_FIELDS if c != "age" and c in df.columns]
    for col in str_fields:
        # 将 NaN 替换为 None，避免 astype(str) 变成 "nan" 字符串
        mask = df[col].isna()
        df.loc[mask, col] = None

    # 5. 构造 persona_id
    df["persona_id"] = df["uuid"].astype(str)

    # 6. 构造 persona_text - 使用 numpy 向量化字符串拼接
    text_cols = []
    for col in FIELDS_PERSONA:
        if col in df.columns:
            text_cols.append(col)
    for col in FIELDS_DEMOGRAPHIC:
        if col in df.columns and col not in ["uuid"]:
            text_cols.append(col)

    # 将所有列转字符串后用 numpy 的 char.add 拼接
    # 比 agg(axis=1) 和列表推导快得多
    df_str = df[text_cols].fillna("").astype(str)
    separator = " | "
    df["persona_text"] = df_str[text_cols[0]]
    for col in text_cols[1:]:
        df["persona_text"] = df["persona_text"] + separator + df_str[col]

    # 7. 过滤：必须至少有 uuid
    df = df.dropna(subset=["uuid"])

    # 8. 按 uuid 排序
    df = df.sort_values("uuid").reset_index(drop=True)

    return df


def save_clean_data(df, path="data/processed/df_full.parquet"):
    """
    保存清洗后的数据

    参数:
        df: 清洗后的 DataFrame
        path: 保存路径
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_parquet(path, index=False)
    print(f"  已保存: {path} ({len(df):,} 行, {len(df.columns)} 列)")
