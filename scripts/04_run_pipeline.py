"""
完整流程脚本

功能：串联所有模块，执行完整的查询流程
运行：python scripts/04_run_pipeline.py
"""

import sys
import os
from datetime import datetime

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from src.utils import Config
from src.retriever import load_index, search_similar, create_database_adapter
from src.llm_client import LLMClient, build_batch_prompts, parse_response


# ============================================
# 用户配置区域 - 在这里修改问题
# ============================================

QUESTION = "退休年龄推迟到64岁，你怎么看？"
K = 10000  # 检索数量

# ============================================


def main():
    """
    主函数：执行完整流程
    """
    print("=" * 60)
    print("法国政策模拟器 - 完整流程")
    print("=" * 60)
    
    print(f"\n问题: {QUESTION}")
    print(f"检索数量: {K}")
    
    # 1. 检查API Key
    if not Config.LLM_API_KEY:
        print("\n[错误] 未找到LLM_API_KEY")
        print("请在 .env 文件中设置 LLM_API_KEY=your_key")
        return
    
    # 2. 加载数据
    print("\n[1/5] 加载数据...")
    df = pd.read_parquet(Config.DATA_PATH)
    print(f"      加载完成：{len(df):,} 行")
    
    # 3. 加载索引和模型
    print("\n[2/5] 加载检索索引...")
    index = load_index(Config.INDEX_PATH)
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer('all-MiniLM-L6-v2')
    print(f"      加载完成")
    
    # 4. 检索相似人物
    print("\n[3/5] 检索相似人物...")
    results_df = search_similar(QUESTION, model, index, df, k=K)
    print(f"      检索完成：{len(results_df)} 条")
    
    # 5. 调用LLM
    print("\n[4/5] 调用LLM获取回答...")
    personas = results_df.to_dict('records')
    prompts = build_batch_prompts(personas, QUESTION)
    
    client = LLMClient(
        api_key=Config.LLM_API_KEY,
        model=Config.LLM_MODEL,
        base_url=Config.LLM_BASE_URL
    )
    llm_responses = client.call_batch_sync(prompts)
    
    # 将回答合并到DataFrame
    response_dict = {r['persona_id']: r['response'] for r in llm_responses if r and r.get("success") and r.get("response")}
    results_df['llm_response'] = results_df['persona_id'].map(response_dict)

    # 仅解析成功回复，失败请求不参与统计
    mask = results_df['llm_response'].notna() & (results_df['llm_response'] != '')
    parsed = results_df.loc[mask, 'llm_response'].apply(parse_response)
    results_df["support_score"] = None
    results_df["stance"] = None
    results_df.loc[mask, 'support_score'] = parsed.apply(lambda x: x['support_score'])
    results_df.loc[mask, 'stance'] = parsed.apply(lambda x: x['stance'])
    print(f"      调用完成")

    # 6. 保存到数据库
    print("\n[5/5] 保存结果到数据库...")
    db = create_database_adapter()
    query_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    db.save(query_id, QUESTION, results_df)
    db.close()
    print(f"      保存完成，Query ID: {query_id}")

    # 7. 输出统计（仅成功请求）
    valid_df = results_df[results_df["stance"].notna()]
    print("\n" + "=" * 60)
    print("统计结果")
    print("=" * 60)
    print(f"总请求数: {len(results_df)}")
    print(f"成功回答数: {len(valid_df)}")
    print(f"失败数: {len(results_df) - len(valid_df)}")
    if len(valid_df) > 0:
        print(f"平均支持度: {valid_df['support_score'].mean():.2f}")

        # 立场分布
        stance_counts = valid_df["stance"].value_counts()
        print("\n立场分布:")
        stance_labels = {"oppose": "反对", "neutral": "中立", "support": "支持"}
        for stance, count in stance_counts.items():
            label = stance_labels.get(stance, stance)
            print(f"  {label}: {count} ({count/len(valid_df)*100:.1f}%)")

        print("\n按职业分组:")
        if "occupation" in valid_df.columns:
            prof_stats = valid_df.groupby("occupation")["support_score"].mean().sort_values(ascending=False)
            for prof, score in prof_stats.head(10).items():
                print(f"  {prof}: {score:.2f}")

        print("\n按省份分组(Top 10):")
        if "departement" in valid_df.columns:
            dept_stats = valid_df.groupby("departement")["support_score"].mean().sort_values(ascending=False)
            for dept, score in dept_stats.head(10).items():
                print(f"  {dept}: {score:.2f}")
    else:
        print("无有效回答，所有请求均失败")
    
    print("\n" + "=" * 60)
    print("流程完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()