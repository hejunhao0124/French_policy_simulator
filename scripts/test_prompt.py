"""
Prompt 模块测试
"""
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.llm_client import build_prompt, build_batch_prompts

# 模拟一条完整数据
TEST_PERSONA = {
    "uuid": "beb99f674",
    "age": 39,
    "sex": "Femme",
    "occupation": "Ouvriers",
    "education_level": "CAP ou BEP",
    "marital_status": "Pacsé(e)",
    "household_type": "Couple avec enfants",
    "departement": "Pas-de-Calais",
    "commune": "Bruay-la-Buissière",
    "persona": "Personne simple et authentique, proche de ses proches",
    "cultural_background": "Née dans le Nord, fière de ses origines",
    "professional_persona": "Travailleuse dans le secteur industriel",
    "sports_persona": "Marche en famille le weekend",
    "arts_persona": "Aime la musique populaire",
    "travel_persona": "Peu de voyages, privilégie les sorties locales",
    "culinary_persona": "Cuisine simple et traditionnelle",
    "career_goals_and_ambitions": "Stabilité professionnelle",
    "skills_and_expertise": "Travail en équipe, gestion de stock",
    "hobbies_and_interests": "Passer du temps avec la famille",
}

QUESTION = "你对退休年龄推迟到64岁有什么看法？"


def test_build_prompt():
    print("=" * 60)
    print("测试 build_prompt")
    print("=" * 60)

    prompt = build_prompt(TEST_PERSONA, QUESTION)
    print(prompt)
    print("=" * 60)


def test_build_prompt_minimal():
    print("=" * 60)
    print("测试 build_prompt（最少字段）")
    print("=" * 60)

    minimal_persona = {
        "persona_id": "test_001",
        "age": 45,
        "occupation": "工人",
    }
    prompt = build_prompt(minimal_persona, QUESTION)
    print(prompt)
    print("=" * 60)


def test_build_batch_prompts():
    print("=" * 60)
    print("测试 build_batch_prompts")
    print("=" * 60)

    personas = [
        TEST_PERSONA,
        {
            "uuid": "a894f0805",
            "age": 57,
            "sex": "Femme",
            "occupation": "Employés",
            "education_level": "Bac+2",
            "marital_status": "Marié(e)",
            "departement": "Loire-Atlantique",
            "commune": "Donges",
            "persona": "Personne organisée et prévoyante",
            "cultural_background": "Originaire de l'Ouest",
        },
        {
            "uuid": "b8b1e903",
            "age": 43,
            "sex": "Homme",
            "occupation": "Ouvriers",
            "education_level": "Baccalauré",
            "marital_status": "Concubinage",
            "departement": "Rhône",
            "commune": "Coise",
        },
    ]

    prompts = build_batch_prompts(personas, QUESTION)
    print(f"生成 {len(prompts)} 条 prompt\n")
    for i, p in enumerate(prompts):
        print(f"--- Prompt {i+1} (id: {p['persona_id']}) ---")
        print(p["prompt"])
        print()


if __name__ == "__main__":
    test_build_prompt()
    test_build_prompt_minimal()
    test_build_batch_prompts()
    print("所有测试完成！")
