"""从论文中提取知识并写入图谱"""
from tools import llm_client
from knowledge.graph import KnowledgeGraph


def populate_graph(kg: KnowledgeGraph, papers: list[dict], query: str):
    """
    从论文列表中提取概念、方法、发现，填充知识图谱。
    同时建立论文-概念、论文-方法的关联。
    """
    # 1. 添加所有论文到图谱
    paper_nodes = []
    for p in papers:
        if not p.get("paper_id"):
            continue
        nid = kg.add_paper(p)
        paper_nodes.append((nid, p))

    # 2. 批量提取概念和方法
    extraction = _extract_batch(papers, query)
    concepts = extraction.get("concepts", [])
    methods = extraction.get("methods", [])
    findings = extraction.get("findings", [])

    # 3. 添加概念节点并建立关联
    for concept in concepts:
        cid = kg.add_concept(concept)
        # 将概念关联到所有相关论文
        for nid, paper in paper_nodes:
            if _is_concept_in_paper(concept, paper):
                kg.add_relation(nid, cid, "about")

    # 4. 添加方法节点
    for method in methods:
        mid = kg.add_method(method)
        for nid, paper in paper_nodes:
            if _is_method_in_paper(method, paper):
                kg.add_relation(nid, mid, "uses")

    # 5. 添加发现节点
    for f in findings:
        paper_title = f.get("paper", "")
        finding_text = f.get("finding", "")
        if not finding_text:
            continue
        # 找到对应的论文节点
        for nid, paper in paper_nodes:
            if paper_title and paper["title"][:30] in paper_title[:30]:
                kg.add_finding(nid, finding_text)
                break

    return extraction


def _extract_batch(papers: list[dict], query: str) -> dict:
    """批量提取概念、方法、发现"""
    abstracts = ""
    for i, p in enumerate(papers[:12]):
        abstract = p.get("abstract", "无摘要") or "无摘要"
        abstracts += f"\n[{i+1}] {p['title']} ({p.get('year', '?')})\n{abstract}\n"

    system_prompt = """你是一位学术研究分析专家。从论文摘要中提取以下信息：

1. **核心概念**（5-10个）：该领域最关键的技术概念
2. **核心方法**（3-8个）：使用的主要技术方法/框架
3. **关键发现**（每篇论文1个）：论文的核心结论

返回 JSON：
{
    "concepts": ["概念1", ...],
    "methods": ["方法1", ...],
    "findings": [{"paper": "论文标题的一部分", "finding": "核心结论"}]
}"""

    user_prompt = f"研究主题：{query}\n\n论文摘要：\n{abstracts}"

    try:
        return llm_client.chat_json(system_prompt, user_prompt)
    except Exception:
        return {"concepts": [], "methods": [], "findings": []}


def _is_concept_in_paper(concept: str, paper: dict) -> bool:
    """简单检查概念是否在论文标题/摘要中出现"""
    text = ((paper.get("title") or "") + " " + (paper.get("abstract") or "")).lower()
    return concept.lower() in text


def _is_method_in_paper(method: str, paper: dict) -> bool:
    """简单检查方法是否在论文中出现"""
    text = ((paper.get("title") or "") + " " + (paper.get("abstract") or "")).lower()
    return method.lower() in text
