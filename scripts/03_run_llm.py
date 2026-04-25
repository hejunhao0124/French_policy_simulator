"""
LLM调用测试脚本

功能：测试LLM客户端是否正常工作
运行：python scripts/03_run_llm.py
"""

import sys
import os

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.llm_client import LLMClient, build_prompt
from src.utils import Config


# ============================================
# 用户配置区域 - 在这里修改问题和人物
# ============================================

QUESTION = "退休年龄推迟到64岁，你怎么看？"

TEST_PERSONA = {
    "persona_id": "test_001",
    "age": 45,
    "profession_category": "工人",
    "department": "59",
    "education_level": "CAP"
}

# ============================================


def main():
    """
    主函数：测试LLM调用
    """
    print("=" * 50)
    print("测试LLM调用")
    print("=" * 50)
    
    print(f"\n当前问题: {QUESTION}")
    print(f"测试人物: {TEST_PERSONA['age']}岁 {TEST_PERSONA['profession_category']}")
    
    # 检查API Key
    if not Config.LLM_API_KEY:
        print("\n[错误] 未找到LLM_API_KEY")
        print("请在 .env 文件中设置 LLM_API_KEY=your_key")
        return

    # 1. 构建Prompt
    print("\n[1/2] 构建Prompt...")
    prompt = build_prompt(TEST_PERSONA, QUESTION)
    print(f"      Prompt预览: {prompt[:80]}...")

    # 2. 调用LLM
    print("\n[2/2] 调用LLM...")
    client = LLMClient(
        api_key=Config.LLM_API_KEY,
        model=Config.LLM_MODEL,
        base_url=Config.LLM_BASE_URL
    )
    
    prompts = [{"persona_id": "test_001", "prompt": prompt}]
    results = client.call_batch_sync(prompts)
    
    # 3. 输出结果
    print("\n" + "=" * 50)
    print("LLM回答:")
    print("=" * 50)
    if results and results[0].get('success'):
        print(results[0].get('response'))
    else:
        print("调用失败:", results[0].get('error', '未知错误'))
    print("=" * 50)


if __name__ == "__main__":
    main()