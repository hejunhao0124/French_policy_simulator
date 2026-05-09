"""
Prompt 模板模块（优化版）

改进：
1. 更清晰的指令，移除「避免中立」与「理性客观」的矛盾
2. 增强 few-shot 示例，帮助 LLM 理解输出格式
3. 明确引导 LLM 基于人物背景表达有依据的倾向
"""


def _get(persona, key):
    """安全获取字段值"""
    val = persona.get(key)
    if val and str(val).strip():
        return str(val).strip()
    return None


def build_prompt(persona, question):
    """
    构建 Prompt — 让 LLM 扮演一位法国公民并回答政策问题

    参数:
        persona: 人物画像字典
        question: 用户问题

    返回:
        str: 完整的 Prompt 字符串
    """
    sections = {}

    # 基础信息
    basic = []
    for key, label in [("age", "年龄"), ("sex", "性别"),
                       ("marital_status", "婚姻"), ("household_type", "家庭")]:
        val = _get(persona, key)
        if val:
            basic.append(f"{label}: {val}")
    if basic:
        sections["基础信息"] = "\n".join(basic)

    # 社会背景
    social = []
    for key, label in [("occupation", "职业"), ("education_level", "学历"),
                       ("departement", "省份"), ("commune", "市镇")]:
        val = _get(persona, key)
        if val:
            social.append(f"{label}: {val}")
    if social:
        sections["社会背景"] = "\n".join(social)

    # 个人画像
    profile = []
    for key, label in [("persona", "性格特质"),
                       ("cultural_background", "文化背景"),
                       ("professional_persona", "职业画像")]:
        val = _get(persona, key)
        if val:
            profile.append(f"{label}: {val}")
    if profile:
        sections["个人画像"] = "\n".join(profile)

    # 兴趣爱好
    hobbies = []
    for key, label in [("sports_persona", "体育"), ("arts_persona", "艺术"),
                       ("travel_persona", "旅行"), ("culinary_persona", "美食"),
                       ("hobbies_and_interests", "兴趣爱好")]:
        val = _get(persona, key)
        if val:
            hobbies.append(f"{label}: {val}")
    if hobbies:
        sections["兴趣爱好"] = "\n".join(hobbies)

    # 职业发展
    career = []
    for key, label in [("career_goals_and_ambitions", "工作目标"),
                       ("skills_and_expertise", "技能专长")]:
        val = _get(persona, key)
        if val:
            career.append(f"{label}: {val}")
    if career:
        sections["职业发展"] = "\n".join(career)

    # 组装人物画像
    persona_parts = []
    for section_name, section_content in sections.items():
        persona_parts.append(f"【{section_name}】\n{section_content}")
    persona_text = "\n\n".join(persona_parts)

    prompt = f"""你是一位法国公民。请根据你的个人画像，回答以下政策问题。请结合你的社会背景（职业、学历、地区）和价值观，表达真实、自然的个人看法，并明确给出你的支持程度。

{persona_text}

【政策问题】
{question}

【回答要求】
1. 第一行（单独一行）必须标注：支持度：X/10（X 为 0 到 10 的整数，0=强烈反对，5=无所谓，10=强烈支持）
2. 后续段落：使用第一人称「我」的视角，表达你的个人看法
3. 语言：中文，简洁精炼（100-200字以内）
4. 必须结合你的人物画像来阐述理由（不要泛泛而谈）
5. 明确表达你的态度倾向，给出有依据的判断

【示例1 - 反对倾向】
支持度：2/10
我今年 52 岁，是北部省的一名工厂工人，学历只有职业证书。退休年龄推迟对我来说就是雪上加霜——我的工作对体力要求很高，每天腰酸背痛。我觉得政府应该考虑不同职业的差异性，而不是一刀切地延长所有人的工作年限。

【示例2 - 支持倾向】
支持度：8/10
我今年 35 岁，是巴黎的一名软件工程师，硕士学历。我认为推迟退休年龄在人口老龄化背景下是合理的——法国的养老金体系需要可持续。而且我的工作性质决定了即使到 64 岁我也能胜任。不过政府应该将省下的养老金用于改善医疗和养老设施。

现在，请以你的人物身份回答："""

    return prompt


def build_batch_prompts(personas, question):
    """
    批量构建 Prompts

    参数:
        personas: 人物画像列表
        question: 用户问题

    返回:
        list[dict]: 每个元素包含 persona_id 和 prompt
    """
    prompts = []
    for persona in personas:
        persona_id = persona.get("persona_id", persona.get("uuid", ""))
        prompt = build_prompt(persona, question)
        prompts.append({
            "persona_id": persona_id,
            "prompt": prompt,
        })
    return prompts
