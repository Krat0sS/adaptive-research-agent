"""知识图谱 — 基于 NetworkX 的学术知识图谱"""
import json
import networkx as nx
from typing import Optional


class KnowledgeGraph:
    """学术知识图谱：论文、概念、方法、发现、空白"""

    def __init__(self):
        self.graph = nx.DiGraph()
        self._paper_count = 0
        self._concept_count = 0
        self._gap_count = 0

    def add_paper(self, paper: dict) -> str:
        """添加论文节点"""
        node_id = f"paper:{paper['paper_id'][:12]}"
        self.graph.add_node(node_id, **{
            "type": "paper",
            "title": paper["title"],
            "authors": ", ".join(paper.get("authors", [])[:3]),
            "year": paper.get("year"),
            "venue": paper.get("venue", ""),
            "abstract": (paper.get("abstract") or "")[:500],
            "citations": paper.get("citation_count", 0)
        })
        self._paper_count += 1
        return node_id

    def add_concept(self, concept: str) -> str:
        """添加概念节点（自动去重）"""
        node_id = f"concept:{concept.lower().replace(' ', '_')}"
        if not self.graph.has_node(node_id):
            self.graph.add_node(node_id, type="concept", name=concept)
            self._concept_count += 1
        return node_id

    def add_method(self, method: str) -> str:
        """添加方法节点"""
        node_id = f"method:{method.lower().replace(' ', '_')}"
        if not self.graph.has_node(node_id):
            self.graph.add_node(node_id, type="method", name=method)
        return node_id

    def add_finding(self, paper_id: str, finding_text: str) -> str:
        """添加发现节点"""
        finding_hash = hash(finding_text) % 100000
        node_id = f"finding:{finding_hash}"
        self.graph.add_node(node_id, type="finding", text=finding_text)
        self.graph.add_edge(paper_id, node_id, relation="produces")
        return node_id

    def add_gap(self, description: str, related_concepts: list[str] = None) -> str:
        """添加研究空白节点"""
        self._gap_count += 1
        node_id = f"gap:{self._gap_count}"
        self.graph.add_node(node_id, type="gap", description=description)
        if related_concepts:
            for c in related_concepts:
                cid = self.add_concept(c)
                self.graph.add_edge(node_id, cid, relation="related_to")
        return node_id

    def add_relation(self, source_id: str, target_id: str, relation: str):
        """添加关系边"""
        if self.graph.has_node(source_id) and self.graph.has_node(target_id):
            self.graph.add_edge(source_id, target_id, relation=relation)

    def get_papers(self) -> list[dict]:
        """获取所有论文节点"""
        return [
            {"id": n, **self.graph.nodes[n]}
            for n in self.graph.nodes
            if self.graph.nodes[n].get("type") == "paper"
        ]

    def get_concepts(self) -> list[dict]:
        """获取所有概念节点"""
        return [
            {"id": n, **self.graph.nodes[n]}
            for n in self.graph.nodes
            if self.graph.nodes[n].get("type") == "concept"
        ]

    def get_gaps(self) -> list[dict]:
        """获取所有研究空白"""
        return [
            {"id": n, **self.graph.nodes[n]}
            for n in self.graph.nodes
            if self.graph.nodes[n].get("type") == "gap"
        ]

    def get_paper_concepts(self, paper_id: str) -> list[str]:
        """获取某篇论文关联的概念"""
        concepts = []
        for _, target, data in self.graph.out_edges(paper_id, data=True):
            if data.get("relation") == "about" and self.graph.nodes[target].get("type") == "concept":
                concepts.append(self.graph.nodes[target]["name"])
        return concepts

    def get_concept_papers(self, concept: str) -> list[str]:
        """获取讨论某概念的所有论文"""
        cid = f"concept:{concept.lower().replace(' ', '_')}"
        if not self.graph.has_node(cid):
            return []
        papers = []
        for source, _, data in self.graph.in_edges(cid, data=True):
            if data.get("relation") == "about":
                papers.append(self.graph.nodes[source].get("title", source))
        return papers

    def get_summary(self) -> dict:
        """图谱摘要统计"""
        types = {}
        for n in self.graph.nodes:
            t = self.graph.nodes[n].get("type", "unknown")
            types[t] = types.get(t, 0) + 1
        return {
            "total_nodes": self.graph.number_of_nodes(),
            "total_edges": self.graph.number_of_edges(),
            "node_types": types
        }

    def to_json(self) -> str:
        """序列化为 JSON"""
        data = nx.node_link_data(self.graph)
        return json.dumps(data, ensure_ascii=False, indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> "KnowledgeGraph":
        """从 JSON 反序列化"""
        kg = cls()
        data = json.loads(json_str)
        kg.graph = nx.node_link_graph(data)
        return kg

    def visualize_text(self) -> str:
        """生成文本格式的图谱摘要"""
        lines = ["═══ 知识图谱概览 ═══\n"]

        papers = self.get_papers()
        concepts = self.get_concepts()
        gaps = self.get_gaps()

        lines.append(f"📄 论文: {len(papers)} 篇")
        lines.append(f"💡 概念: {len(concepts)} 个")
        lines.append(f"🔍 研究空白: {len(gaps)} 个")
        lines.append(f"🔗 关系边: {self.graph.number_of_edges()} 条\n")

        if concepts:
            lines.append("── 核心概念 ──")
            for c in concepts[:10]:
                related_papers = self.get_concept_papers(c["name"])
                lines.append(f"  • {c['name']} ({len(related_papers)} 篇论文)")

        if gaps:
            lines.append("\n── 研究空白 ──")
            for g in gaps[:5]:
                lines.append(f"  ⚠ {g['description']}")

        return "\n".join(lines)
