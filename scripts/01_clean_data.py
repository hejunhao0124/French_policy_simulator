"""
数据清洗脚本

功能：加载原始数据 -> 清洗 -> 保存到 data/processed/ -> EDA 统计
运行：python scripts/01_clean_data.py
"""

import sys
import os

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_pipeline import (
    load_full_data, clean_data, save_clean_data,
    print_summary_stats
)
from src.utils import Config


# ============================================
# 配置区域
# ============================================

RUN_EDA = True  # 是否运行 EDA 统计分析
OUTPUT_DIR = "data/processed"
SAVE_PATH = os.path.join(OUTPUT_DIR, "df_full.parquet")

# ============================================


def main():
    """
    主函数：执行完整的数据清洗流程
    """
    print("=" * 60)
    print("法国人物画像 - 数据清洗")
    print("=" * 60)

    # 1. 加载数据
    print("\n[1/3] 加载原始数据...")
    print(f"      数据集: {Config.DATASET_NAME} (流式采样 100k)")
    df = load_full_data(sample_size=100000, seed=42)
    print(f"      加载完成：{len(df):,} 行，{len(df.columns)} 列")

    # 2. 清洗数据
    print("\n[2/3] 清洗数据...")
    df_clean = clean_data(df)
    print(f"      清洗完成：{len(df_clean):,} 行")

    # 3. 保存数据
    print("\n[3/3] 保存数据...")
    save_clean_data(df_clean, SAVE_PATH)
    print(f"      保存完成")

    # 4. EDA 统计
    if RUN_EDA:
        print("\n" + "-" * 60)
        print("EDA 统计分析")
        print("-" * 60)
        print_summary_stats(df_clean)

    print("\n" + "=" * 60)
    print("数据清洗完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()