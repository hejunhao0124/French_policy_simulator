"""
响应解析模块（优化版）

改进：
1. 区分「解析成功/失败」，不再静默给默认值
2. 支持更多输出格式
3. 返回 parse_success 标记，让上游了解哪些回复未被正确解析
"""

import re


def parse_response(response):
    """
    解析 LLM 回答，提取支持度分数和立场

    参数:
        response: LLM 原始回答文本

    返回:
        dict: {
            full_response:  原始文本
            support_score:  支持度 (0.0-1.0)
            stance:         立场 (oppose/neutral/support)
            raw_score:      原始整数分数 (0-10)，None 表示未提取到
            parse_success:  True=从回复中成功提取分数，False=使用默认值
        }
    """
    full_response = response
    raw_score = None

    # 按优先级依次匹配多种格式
    patterns = [
        # 模式1: 支持度：X/10 或 支持度: X/10（最精确）
        r'支持度[：:]\s*(\d+)(?:\s*/\s*10|\s*分)?',
        # 模式2: 支持度 X/10（无冒号）
        r'支持度\s*(\d+)\s*/\s*10',
        # 模式3: 支持度评分[：:]\s*(\d+)
        r'支持度评分[：:]\s*(\d+)',
        # 模式4: 英文 support score: X/10
        r'support\s*score[：:]\s*(\d+)',
        # 模式5: 任何 X/10 格式（最后兜底）
        r'(\d+)\s*/\s*10',
    ]

    for pattern in patterns:
        match = re.search(pattern, response, re.IGNORECASE)
        if match:
            raw_score = int(match.group(1))
            break

    # 计算归一化分数和立场
    if raw_score is not None:
        support_score = raw_score / 10.0
        parse_success = True
    else:
        # 无法提取分数：使用默认中立值并标记解析失败
        support_score = 0.5
        parse_success = False

    # 确保分数在 0-1 范围内
    support_score = max(0.0, min(1.0, support_score))

    # 判断立场
    if support_score < 0.34:
        stance = "oppose"
    elif support_score < 0.67:
        stance = "neutral"
    else:
        stance = "support"

    return {
        "full_response": full_response,
        "support_score": support_score,
        "stance": stance,
        "raw_score": raw_score,
        "parse_success": parse_success,
    }
