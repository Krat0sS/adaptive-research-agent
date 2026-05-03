"""论文解析工具 — 从论文元数据中提取结构化信息"""
from tools import llm_client


def extract_concepts(papers: list[dict]) -> dict:
    """
    从一批论文中提取核心概念、方法和发现。
    
    Returns:
        {
            "concepts": ["概念1", "概念2", ...],
            "methods": ["方法1", "方法2", ...],
            "findings": [{"paper": "标题", "finding": "发现内容"}, ...]
        }
    """
    # 构建论文摘要文本
    abstracts_text = ""
    for i, p in enumerate(papers[:15]):  # 最多处理 15 篇，避免 token 爆炸
        abstract = p.get("abstract", "无摘要")
        abstracts_text += f"\n--- 论文 {i+1}: {p['title']} ({p.get('year', '?')}) ---\n{abstract}\n"

    system_prompt = """你是一位学术研究分析专家。请从给定的论文摘要中提取：

1. **核心概念**（concepts）：该领域最重要的 5-10 个技术概念/术语
2. **核心方法**（methods）：论文中使用的主要技术方法/框架/算法
3. **关键发现**（findings）：每篇论文最核心的结论或贡献（保持原文精度，不要泛化）

返回 JSON 格式：
{
    "concepts": ["概念1", "概念2", ...],
    "methods": ["方法1", "方法2", ...],
    "findings": [
        {"paper": "论文标题", "finding": "具体发现"},
        ...
    ]
}"""

    return llm_client.chat_json(system_prompt, f"以下是待分析的论文摘要：\n{abstracts_text}")


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
