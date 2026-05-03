"""单元测试 — 知识图谱"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from knowledge.graph import KnowledgeGraph


def test_add_paper():
    """测试添加论文"""
    kg = KnowledgeGraph()
    paper = {
        "paper_id": "abc123",
        "title": "Test Paper",
        "authors": ["Alice", "Bob"],
        "year": 2024,
        "venue": "NeurIPS",
        "abstract": "This is a test abstract.",
        "citation_count": 10
    }
    nid = kg.add_paper(paper)
    assert nid == "paper:abc123"
    papers = kg.get_papers()
    assert len(papers) == 1
    assert papers[0]["title"] == "Test Paper"
    print("✓ test_add_paper passed")


def test_concept_dedup():
    """测试概念去重"""
    kg = KnowledgeGraph()
    c1 = kg.add_concept("Reinforcement Learning")
    c2 = kg.add_concept("Reinforcement Learning")
    assert c1 == c2, "Duplicate concepts should have same ID"
    assert len(kg.get_concepts()) == 1
    print("✓ test_concept_dedup passed")


def test_relations():
    """测试关系建立"""
    kg = KnowledgeGraph()
    pid = kg.add_paper({
        "paper_id": "test1",
        "title": "RL Paper",
        "authors": [],
        "year": 2024,
        "venue": "",
        "abstract": "About reinforcement learning.",
        "citation_count": 0
    })
    cid = kg.add_concept("Reinforcement Learning")
    kg.add_relation(pid, cid, "about")

    related = kg.get_paper_concepts(pid)
    assert "Reinforcement Learning" in related
    print("✓ test_relations passed")


def test_gap_analysis():
    """测试研究空白添加"""
    kg = KnowledgeGraph()
    gid = kg.add_gap("No work on multi-agent RL in real traffic", ["MARL", "traffic"])
    gaps = kg.get_gaps()
    assert len(gaps) == 1
    assert "multi-agent" in gaps[0]["description"]
    print("✓ test_gap_analysis passed")


def test_serialization():
    """测试 JSON 序列化/反序列化"""
    kg = KnowledgeGraph()
    kg.add_paper({
        "paper_id": "serial_test",
        "title": "Serialization Test",
        "authors": ["Test"],
        "year": 2024,
        "venue": "ICML",
        "abstract": "Test abstract.",
        "citation_count": 5
    })
    kg.add_concept("Test Concept")

    # 序列化
    json_str = kg.to_json()
    assert len(json_str) > 100

    # 反序列化
    kg2 = KnowledgeGraph.from_json(json_str)
    assert len(kg2.get_papers()) == 1
    assert kg2.get_papers()[0]["title"] == "Serialization Test"
    assert len(kg2.get_concepts()) == 1
    print("✓ test_serialization passed")


def test_summary():
    """测试图谱摘要"""
    kg = KnowledgeGraph()
    kg.add_paper({"paper_id": "p1", "title": "P1", "authors": [], "year": 2024, "venue": "", "abstract": "", "citation_count": 0})
    kg.add_paper({"paper_id": "p2", "title": "P2", "authors": [], "year": 2024, "venue": "", "abstract": "", "citation_count": 0})
    kg.add_concept("C1")

    summary = kg.get_summary()
    assert summary["node_types"]["paper"] == 2
    assert summary["node_types"]["concept"] == 1
    print("✓ test_summary passed")


if __name__ == "__main__":
    test_add_paper()
    test_concept_dedup()
    test_relations()
    test_gap_analysis()
    test_serialization()
    test_summary()
    print("\n所有知识图谱测试通过 ✓")
