"""论文解析工具 — 从论文元数据中提取结构化信息

注意：extract_concepts 功能已整合到 knowledge/extractor._extract_batch，
此处保留接口兼容性，内部委托给 extractor。
"""
from tools import llm_client


def extract_concepts(papers: list[dict]) -> dict:
    """
    从一批论文中提取核心概念、方法和发现。
    
    ⚠️ 已废弃：此函数与 knowledge/extractor._extract_batch 功能重复。
    保留仅为向后兼容。新代码应直接使用 extractor._extract_batch()。
    
    Returns:
        {
            "concepts": ["概念1", "概念2", ...],
            "methods": ["方法1", "方法2", ...],
            "findings": [{"paper": "标题", "finding": "发现内容"}, ...]
        }
    """
    # 延迟导入避免循环依赖
    from knowledge import extractor as _extractor
    from knowledge.graph import KnowledgeGraph

    # 委托给 extractor._extract_batch，用空 query 兜底
    return _extractor._extract_batch(papers, query="学术研究")


def assess_paper_relevance(paper: dict, query: str) -> dict:
    """评估单篇论文与查询主题的相关性"""
    system_prompt = """你是一位学术文献评估专家。请评估给定论文与查询主题的相关性。

返回 JSON：
{
    "relevance_score": 0.0-1.0,
    "reason": "简短说明为什么相关/不相关",
    "key_contribution": "该论文对查询主题的核心贡献（一句话）"
}"""

    user_prompt = f"查询主题：{query}\n\n论文标题：{paper['title']}\n摘要：{paper.get('abstract', '无摘要')}"

    return llm_client.chat_json(system_prompt, user_prompt)
