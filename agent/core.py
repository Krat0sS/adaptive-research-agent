"""Agent 核心主循环 — 三层环路"""
import time
import uuid
from typing import Optional

from tools import search, paper_parser, llm_client
from knowledge.graph import KnowledgeGraph
from knowledge import extractor, gap_analyzer
from learning.user_model import UserModel
from learning.bandit import ThompsonSamplingBandit, generate_arm_id
from learning.feedback import FeedbackCollector
from output import proposal as proposal_gen
import config


class ResearchAgent:
    """
    自适应研究助手 Agent。
    
    三层环路：
    1. 反应层：搜索、提取、格式化
    2. 慎思层：图谱推理、空白识别、提案生成
    3. 元推理层：停止规则、资源分配、用户模型查询
    """

    def __init__(self):
        self.kg = KnowledgeGraph()
        self.user_model = UserModel()
        self.bandit = ThompsonSamplingBandit()
        self.feedback = FeedbackCollector()
        self.current_task_id = None
        self.search_round = 0
        self.collected_papers = []
        self._last_strategy = None  # 记录最后一次策略，供隐式反馈使用
        self._task_start_time = None  # 记录任务开始时间

    def run(self, query: str, verbose: bool = True) -> dict:
        """
        主入口：给定研究主题，返回完整的研究提案。
        
        Args:
            query: 研究主题
            verbose: 是否打印过程信息
        
        Returns:
            {
                "proposal": {...},
                "graph_summary": {...},
                "gaps": [...],
                "search_stats": {...}
            }
        """
        self.current_task_id = str(uuid.uuid4())[:8]
        self.search_round = 0
        self.collected_papers = []
        self.kg = KnowledgeGraph()  # 每次任务重建图谱
        self._task_start_time = time.time()

        # 获取用户偏好
        prefs = self.user_model.get_preferences()
        if verbose:
            print(f"\n{'='*60}")
            print(f"🔬 研究主题: {query}")
            print(f"📋 任务 ID: {self.current_task_id}")
            print(f"👤 历史交互: {len(self.user_model.data.get('interaction_history', []))} 次")
            print(f"{'='*60}\n")

        # ═══ 主循环：搜索 → 提取 → 评估 → 决策 ═══
        while self.search_round < config.MAX_SEARCH_ROUNDS:
            self.search_round += 1

            if verbose:
                print(f"── 搜索轮次 {self.search_round}/{config.MAX_SEARCH_ROUNDS} ──")

            # [元推理] 决定本轮搜索策略
            strategy = self._decide_strategy(query, prefs)
            self._last_strategy = strategy  # 记录供隐式反馈使用
            if verbose:
                print(f"  策略: keywords={strategy['keywords']}, venue={strategy.get('venue', '不限')}")

            # [反应层] 执行搜索
            papers = self._execute_search(query, strategy)
            if verbose:
                print(f"  获取: {len(papers)} 篇论文")

            if not papers or (len(papers) == 1 and "error" in papers[0]):
                if verbose:
                    print(f"  ⚠ 搜索失败，跳过本轮")
                continue

            # [反应层] 提取知识并填充图谱
            extraction = extractor.populate_graph(self.kg, papers, query)
            self.collected_papers.extend(papers)

            if verbose:
                concepts = extraction.get("concepts", [])
                print(f"  提取: {len(concepts)} 个概念, {len(extraction.get('methods', []))} 个方法")

            # [元推理] 评估是否应该停止搜索
            should_stop = self._meta_reasoning_should_stop(papers, extraction)
            if should_stop:
                if verbose:
                    print(f"\n  ✅ 元推理判定：边际信息增益不足，停止搜索")
                break

            if verbose:
                print(f"  ➜ 继续搜索...\n")

        # ═══ 慎思层：分析空白 + 生成提案 ═══
        if verbose:
            print(f"\n── 慎思层：分析研究空白 ──")

        gaps = gap_analyzer.identify_gaps(self.kg, query)
        if verbose:
            for g in gaps:
                print(f"  ⚠ {g['description'][:80]}")

        if verbose:
            print(f"\n── 慎思层：生成研究提案 ──")

        prop = proposal_gen.generate_proposal(self.kg, query, gaps, prefs)

        # 保存反馈请求
        self.feedback.prepare_feedback_request(self.current_task_id, prop, self.collected_papers)

        # 更新搜索策略统计
        self._update_bandit(query, strategy, prop)

        if verbose:
            self._print_proposal(prop)
            print(f"\n── 图谱摘要 ──")
            print(self.kg.visualize_text())

        return {
            "task_id": self.current_task_id,
            "proposal": prop,
            "graph_summary": self.kg.get_summary(),
            "gaps": gaps,
            "search_stats": {
                "rounds": self.search_round,
                "total_papers": len(self.collected_papers),
                "bandit_stats": self.bandit.get_stats()
            }
        }

    def provide_feedback(self, rating: int, comments: str = "") -> dict:
        """用户提供反馈后调用，更新用户模型"""
        fb = self.feedback.collect_explicit_feedback(self.current_task_id, rating, comments)
        self.user_model.record_interaction(
            query="",  # 可从上下文补充
            strategy={"round": self.search_round},
            feedback=fb
        )
        # 用显式评分更新赌博机
        if self._last_strategy:
            from learning.bandit import generate_arm_id
            arm_id = generate_arm_id(
                self._last_strategy.get("keywords", []),
                self._last_strategy.get("venue"),
                self._last_strategy.get("year_range")
            )
            reward = rating / 5.0  # 归一化到 [0, 1]
            self.bandit.update(arm_id, reward)
        return {"status": "feedback_recorded", "data": fb}

    def provide_implicit_feedback(self, user_actions: dict) -> dict:
        """
        基于隐式信号更新赌博机（用户未给显式反馈时的 fallback）。
        
        user_actions 可包含：
          - time_spent: 用户查看提案的时间（秒）
          - copied_text: 用户是否复制了提案文本
          - asked_followup: 用户是否追问了相关问题
          - regenerated: 用户是否要求重新生成
        """
        signals = self.feedback.auto_detect_signals({}, user_actions)

        # 将隐式信号转换为 reward
        reward = 0.5  # 默认中性
        engagement = signals.get("engagement", "low")
        if engagement == "high":
            reward = 0.7
        elif engagement == "low":
            reward = 0.3

        if signals.get("utility") == "high":
            reward = max(reward, 0.75)
        if signals.get("satisfaction") == "low":
            reward = min(reward, 0.25)
        if signals.get("interest") == "high":
            reward = max(reward, 0.65)

        # 更新赌博机
        if self._last_strategy:
            from learning.bandit import generate_arm_id
            arm_id = generate_arm_id(
                self._last_strategy.get("keywords", []),
                self._last_strategy.get("venue"),
                self._last_strategy.get("year_range")
            )
            self.bandit.update(arm_id, reward)

        return {"status": "implicit_feedback_recorded", "signals": signals, "reward": reward}

    # ── 私有方法 ──

    def _decide_strategy(self, query: str, prefs: dict) -> dict:
        """[元推理] 决定本轮搜索策略"""
        # 基于用户偏好和赌博机选择关键词组合
        base_keywords = query.split()[:3]

        # 尝试从赌博机中选择历史上效果好的策略
        known_arms = list(self.bandit.arms.keys())
        if known_arms and self.search_round > 1:
            selected = self.bandit.select_arm(known_arms)
            if selected:
                # 解析臂 ID 恢复策略
                parts = selected.split("|")
                keywords = parts[0].split("+")
                venue = None
                for p in parts[1:]:
                    if p.startswith("v:"):
                        venue = p[2:]
                return {"keywords": keywords, "venue": venue}

        # 默认策略
        strategy = {"keywords": base_keywords}

        # 根据用户偏好添加会议过滤
        venue_weights = prefs.get("venue_weights", {})
        if venue_weights:
            top_venue = max(venue_weights, key=venue_weights.get)
            strategy["venue"] = top_venue

        # 根据 recency_bias 决定年份范围
        recency = prefs.get("recency_bias", 0.7)
        if recency > 0.8:
            strategy["year_range"] = "2022-2026"
        elif recency > 0.5:
            strategy["year_range"] = "2020-2026"

        return strategy

    def _execute_search(self, query: str, strategy: dict) -> list[dict]:
        """[反应层] 执行搜索"""
        keywords = strategy["keywords"]
        search_query = " ".join(keywords)

        papers = search.search_papers(
            query=search_query,
            limit=config.MAX_PAPERS_PER_SEARCH,
            year_range=strategy.get("year_range"),
            venue=strategy.get("venue"),
            min_citations=self.user_model.get_preferences().get("citation_threshold", 0)
        )

        return papers

    def _meta_reasoning_should_stop(self, papers: list[dict], extraction: dict) -> bool:
        """
        [元推理] 判断是否应该停止搜索。
        
        三条规则：
        1. 已收集足够论文
        2. 本轮新概念数量太少（边际信息增益低）— 关键改进：与图谱已有概念做差集，
           只计算真正的"增量概念"，避免重复概念导致误判
        3. 达到最少论文数且已完成至少 2 轮，且新论文与已有图谱高度重叠
        """
        # 规则 1：已收集足够论文
        if len(self.collected_papers) >= config.MAX_PAPERS_PER_SEARCH * config.MAX_SEARCH_ROUNDS * 0.5:
            return True

        # 规则 2：本轮增量概念数量太少（边际信息增益低）
        # 关键改进：将本轮提取的概念与图谱中已有概念做差集
        new_concepts = extraction.get("concepts", [])
        if self.search_round > 1 and len(new_concepts) > 0:
            existing_concepts = {
                c["name"].lower() for c in self.kg.get_concepts()
            }
            truly_new = [c for c in new_concepts if c.lower() not in existing_concepts]
            if len(truly_new) < 2:
                return True
        elif self.search_round > 1 and len(new_concepts) < 2:
            return True

        # 规则 3：达到最少论文数且已完成至少 2 轮
        if len(self.collected_papers) >= config.MIN_PAPERS_FOR_ANALYSIS and self.search_round >= 2:
            # 检查是否有新论文与已有图谱高度重叠
            existing_titles = {p["title"].lower()[:30] for p in self.kg.get_papers()}
            new_count = sum(
                1 for p in papers
                if p.get("title", "").lower()[:30] not in existing_titles
            )
            if new_count < 3:
                return True

        return False

    def _update_bandit(self, query: str, strategy: dict, proposal: dict):
        """注册赌博机臂（不预设 reward，等用户反馈后更新）"""
        arm_id = generate_arm_id(
            strategy["keywords"],
            strategy.get("venue"),
            strategy.get("year_range")
        )
        self.bandit.add_arm(arm_id)
        # 不再预设 0.5 reward —— 等 provide_feedback 或 provide_implicit_feedback 更新

    def _print_proposal(self, prop: dict):
        """打印提案"""
        print(f"\n{'='*60}")
        print(f"📄 研究提案")
        print(f"{'='*60}")
        print(f"\n📌 标题: {prop.get('title', 'N/A')}")
        print(f"\n❓ 研究问题: {prop.get('research_question', 'N/A')}")
        print(f"\n💡 动机:\n{prop.get('motivation', 'N/A')}")
        print(f"\n🔧 方法论:\n{prop.get('methodology', 'N/A')}")
        print(f"\n🎯 预期贡献:")
        for i, c in enumerate(prop.get("expected_contributions", []), 1):
            print(f"  {i}. {c}")
        print(f"\n📚 相关工作总结:\n{prop.get('related_work_summary', 'N/A')}")
        print(f"\n⚡ 开放挑战:")
        for i, c in enumerate(prop.get("open_challenges", []), 1):
            print(f"  {i}. {c}")
        print(f"\n{'='*60}")
