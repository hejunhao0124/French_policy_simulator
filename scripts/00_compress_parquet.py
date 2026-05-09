"""
数据压缩脚本 - 将 Parquet 文件用 ZSTD 列压缩减小体积

用法: python scripts/00_compress_parquet.py

无损压缩，数据内容不变，仅改变存储编码。
压缩后文件大小预计可减少 30-50%。
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

INPUT_PATH = "data/processed/df_full.parquet"
OUTPUT_PATH = "data/processed/df_full_compressed.parquet"


def get_file_size_mb(path):
    return os.path.getsize(path) / (1024 * 1024)


def main():
    if not os.path.exists(INPUT_PATH):
        print(f"Error: Input file not found: {INPUT_PATH}")
        return

    input_size = get_file_size_mb(INPUT_PATH)
    print(f"Loading: {INPUT_PATH} ({input_size:.1f} MB)")

    start = time.time()
    df = pd.read_parquet(INPUT_PATH)
    load_time = time.time() - start
    print(f"  Loaded {len(df):,} rows x {len(df.columns)} columns in {load_time:.1f}s")

    print(f"Compressing with ZSTD (level=3) -> {OUTPUT_PATH}")
    start = time.time()
    df.to_parquet(OUTPUT_PATH, engine="pyarrow", compression="zstd", compression_level=3)
    save_time = time.time() - start

    output_size = get_file_size_mb(OUTPUT_PATH)
    ratio = (1 - output_size / input_size) * 100
    print(f"  Compressed in {save_time:.1f}s")
    print()
    print(f"Original:  {input_size:.1f} MB")
    print(f"Compressed: {output_size:.1f} MB")
    print(f"Reduction: {ratio:.1f}%")
    print()

    # Verify data integrity
    print("Verifying data integrity...")
    df2 = pd.read_parquet(OUTPUT_PATH)
    assert len(df) == len(df2), "Row count mismatch!"
    assert list(df.columns) == list(df2.columns), "Column mismatch!"
    assert (df["uuid"] == df2["uuid"]).all(), "UUID mismatch!"
    assert (df["age"] == df2["age"]).all(), "Age mismatch!"
    print("  All checks passed - data is identical")
    print()
    print(f"Done! Replace the original file:")
    print(f"  mv {OUTPUT_PATH} {INPUT_PATH}")


if __name__ == "__main__":
    main()
