"""研究提案生成器"""
from tools import llm_client
from knowledge.graph import KnowledgeGraph


def generate_proposal(kg: KnowledgeGraph, query: str, gaps: list[dict], user_prefs: dict = None) -> dict:
    """
    基于知识图谱和研究空白，生成结构化研究提案。
    
    Returns:
        {
            "title": "建议的论文标题",
            "research_question": "核心研究问题",
            "motivation": "研究动机（基于空白分析）",
            "methodology": "建议的方法论",
            "expected_contributions": ["贡献1", "贡献2"],
            "related_work_summary": "相关工作总结",
            "open_challenges": ["挑战1", "挑战2"]
        }
    """
    papers = kg.get_papers()
    concepts = kg.get_concepts()

    # 构建相关工作摘要
    related_work = []
    for p in papers[:10]:
        related = kg.get_paper_concepts(p["id"])
        related_work.append(
            f"- {p['title']} ({p.get('year', '?')}, {p.get('venue', '')})\n"
            f"  概念: {', '.join(related[:4])}"
        )

    gap_descriptions = []
    for g in gaps[:4]:
        gap_descriptions.append(f"- {g['description']} (置信度: {g.get('confidence', '?')})")

    concept_names = [c["name"] for c in concepts[:15]]

    # 根据用户偏好调整风格
    style_hint = ""
    if user_prefs:
        depth = user_prefs.get("depth_vs_breadth", 0.5)
        if depth > 0.7:
            style_hint = "提案应深入聚焦，强调方法论的严谨性和技术细节。"
        elif depth < 0.3:
            style_hint = "提案应广泛覆盖，强调跨领域的连接和宏观视角。"

    system_prompt = """你是一位资深学术研究者，擅长撰写高质量的研究提案。

基于提供的文献图谱和研究空白，生成一份结构化的研究提案。
要求：
1. 研究问题必须从已识别的空白中自然推导
2. 方法论必须具体可行，引用已有的相关方法
3. 贡献必须明确、可验证
4. 语言专业但清晰

返回 JSON：
{
    "title": "建议的论文标题",
    "research_question": "核心研究问题（一句话）",
    "motivation": "研究动机（为什么这个问题重要，基于什么空白）",
    "methodology": "建议的方法论（具体步骤）",
    "expected_contributions": ["贡献1", "贡献2", "贡献3"],
    "related_work_summary": "相关工作总结（2-3句话概括已有工作及其局限）",
    "open_challenges": ["可能的挑战1", "可能的挑战2"]
}"""

    user_prompt = f"""研究主题：{query}

{style_hint}

已有文献（共 {len(papers)} 篇）：
{chr(10).join(related_work)}

核心概念：{', '.join(concept_names)}

已识别的研究空白：
{chr(10).join(gap_descriptions) if gap_descriptions else "（尚未识别到明确空白，请基于文献局限性推导）"}"""

    return llm_client.chat_json(system_prompt, user_prompt)
