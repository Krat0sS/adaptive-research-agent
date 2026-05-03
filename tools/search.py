"""学术搜索工具 — 基于 Semantic Scholar API"""
import time
import requests
from typing import Optional
import config

_last_request_time = 0


def _rate_limit():
    """简单的速率限制"""
    global _last_request_time
    elapsed = time.time() - _last_request_time
    if elapsed < config.S2_RATE_LIMIT_DELAY:
        time.sleep(config.S2_RATE_LIMIT_DELAY - elapsed)
    _last_request_time = time.time()


def search_papers(
    query: str,
    limit: int = None,
    year_range: Optional[str] = None,
    venue: Optional[str] = None,
    min_citations: int = 0
) -> list[dict]:
    """
    搜索论文。
    
    Args:
        query: 搜索关键词
        limit: 返回数量上限
        year_range: 年份范围，如 "2020-2025"
        venue: 会议/期刊过滤
        min_citations: 最低引用数
    
    Returns:
        论文列表，每项包含 title, authors, year, venue, abstract, citationCount, paperId
    """
    if limit is None:
        limit = config.MAX_PAPERS_PER_SEARCH

    _rate_limit()

    params = {
        "query": query,
        "limit": min(limit, 100),
        "fields": "title,authors,year,venue,abstract,citationCount,referenceCount,externalIds"
    }

    if year_range:
        params["year"] = year_range

    headers = {}
    if config.S2_API_KEY:
        headers["x-api-key"] = config.S2_API_KEY

    try:
        resp = requests.get(
            f"{config.S2_BASE_URL}/paper/search",
            params=params,
            headers=headers,
            timeout=15
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        return [{"error": str(e)}]

    papers = data.get("data", [])

    # 后过滤
    results = []
    for p in papers:
        if not p.get("title"):
            continue
        if min_citations and (p.get("citationCount") or 0) < min_citations:
            continue
        if venue and venue.lower() not in (p.get("venue") or "").lower():
            continue

        results.append({
            "paper_id": p.get("paperId", ""),
            "title": p.get("title", ""),
            "authors": [a.get("name", "") for a in (p.get("authors") or [])],
            "year": p.get("year"),
            "venue": p.get("venue", ""),
            "abstract": p.get("abstract", ""),
            "citation_count": p.get("citationCount", 0),
            "reference_count": p.get("referenceCount", 0)
        })

    return results


def get_paper_details(paper_id: str) -> dict:
    """获取论文详细信息，包括引用和被引论文"""
    _rate_limit()

    headers = {}
    if config.S2_API_KEY:
        headers["x-api-key"] = config.S2_API_KEY

    fields = "title,authors,year,venue,abstract,citationCount,references.title,references.paperId,citations.title,citations.paperId"

    try:
        resp = requests.get(
            f"{config.S2_BASE_URL}/paper/{paper_id}",
            params={"fields": fields},
            headers=headers,
            timeout=15
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        return {"error": str(e)}


def build_citation_query(concepts: list[str], strategy: dict = None) -> str:
    """
    基于概念列表和策略参数构建搜索查询。
    
    strategy 可包含：
      - focus_terms: 重点术语（权重更高）
      - exclude_terms: 排除术语
      - conjunction: 是否用 AND 连接（默认 OR）
    """
    if strategy is None:
        strategy = {}

    terms = list(concepts)

    focus = strategy.get("focus_terms", [])
    if focus:
        # 重点术语放前面
        terms = focus + [t for t in terms if t not in focus]

    exclude = strategy.get("exclude_terms", [])
    query = " ".join(terms[:5])  # 最多 5 个术语，避免过窄

    if exclude:
        query += " " + " ".join(f"-{e}" for e in exclude[:2])

    return query
