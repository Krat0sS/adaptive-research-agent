"""研究空白分析器"""
from tools import llm_client
from knowledge.graph import KnowledgeGraph


def identify_gaps(kg: KnowledgeGraph, query: str) -> list[dict]:
    """
    基于知识图谱中的论文、概念和发现，推断研究空白。
    
    Returns:
        [{"description": "空白描述", "related_concepts": [...], "confidence": 0.0-1.0}]
    """
    papers = kg.get_papers()
    concepts = kg.get_concepts()

    if len(papers) < 3:
        return []

    # 构建图谱摘要
    paper_summaries = []
    for p in papers[:15]:
        title = p.get("title", "")
        year = p.get("year", "?")
        venue = p.get("venue", "")
        related = kg.get_paper_concepts(p["id"])
        paper_summaries.append(f"- {title} ({year}, {venue}) — 概念: {', '.join(related[:5])}")

    concept_list = [c["name"] for c in concepts[:20]]

    system_prompt = """你是一位资深学术研究者。基于已有的论文和概念，识别该领域的研究空白。

好的研究空白应该是：
1. 从现有论文的局限性中逻辑推导出来的
2. 有明确的研究方向，不是泛泛的"需要更多研究"
3. 具有学术价值和可行性

返回 JSON：
{
    "gaps": [
        {
            "description": "具体的研究空白描述",
            "related_concepts": ["相关概念1", "相关概念2"],
            "reasoning": "为什么这是一个空白（基于哪些论文的什么局限）",
            "confidence": 0.0-1.0
        }
    ]
}

返回 2-4 个最有价值的空白。"""

    user_prompt = f"""研究主题：{query}

已有论文：
{chr(10).join(paper_summaries)}

核心概念：{', '.join(concept_list)}"""

    try:
        result = llm_client.chat_json(system_prompt, user_prompt)
        gaps = result.get("gaps", [])
    except Exception:
        gaps = []

    # 将空白写入图谱
    for gap in gaps:
        nid = kg.add_gap(gap["description"], gap.get("related_concepts", []))
        gap["node_id"] = nid

    return gaps
