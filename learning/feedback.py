"""反馈信号收集与处理"""
from typing import Optional


class FeedbackCollector:
    """收集用户对搜索结果和提案的反馈"""

    def __init__(self):
        self.pending_feedback = {}

    def prepare_feedback_request(self, task_id: str, proposal: dict, papers: list[dict]):
        """准备反馈请求，在提案展示后等待用户评价"""
        self.pending_feedback[task_id] = {
            "proposal": proposal,
            "papers": papers,
            "ratings": {},
            "modifications": None,
            "paper_feedback": {}
        }

    def collect_explicit_feedback(self, task_id: str, rating: int, comments: str = "") -> dict:
        """收集显式反馈（用户直接评分）"""
        if task_id not in self.pending_feedback:
            return {"error": "no pending feedback for this task"}

        fb = self.pending_feedback[task_id]
        fb["ratings"]["overall"] = rating
        fb["comments"] = comments

        return self._process_feedback(task_id)

    def collect_paper_feedback(self, task_id: str, paper_idx: int, relevant: bool):
        """收集论文级别反馈（哪些论文相关/不相关）"""
        if task_id in self.pending_feedback:
            self.pending_feedback[task_id]["paper_feedback"][paper_idx] = relevant

    def collect_modification_feedback(self, task_id: str, modifications: str):
        """收集用户对提案的修改（隐式反馈）"""
        if task_id in self.pending_feedback:
            self.pending_feedback[task_id]["modifications"] = modifications

    def _process_feedback(self, task_id: str) -> dict:
        """处理反馈，生成结构化反馈数据"""
        fb = self.pending_feedback.pop(task_id, {})

        result = {
            "rating": fb.get("ratings", {}).get("overall"),
            "modifications": fb.get("modifications"),
            "comments": fb.get("comments", ""),
            "paper_relevance": {}
        }

        # 计算论文相关率
        paper_fb = fb.get("paper_feedback", {})
        if paper_fb:
            relevant_count = sum(1 for v in paper_fb.values() if v)
            result["paper_relevance_rate"] = relevant_count / len(paper_fb)
        else:
            result["paper_relevance_rate"] = None

        return result

    def auto_detect_signals(self, proposal: dict, user_actions: dict) -> dict:
        """
        自动检测隐式反馈信号（不需要用户主动评分）。
        
        user_actions 可包含：
          - time_spent: 用户查看提案的时间（秒）
          - copied_text: 用户是否复制了提案文本
          - asked_followup: 用户是否追问了相关问题
          - regenerated: 用户是否要求重新生成
        """
        signals = {}

        time_spent = user_actions.get("time_spent", 0)
        if time_spent > 60:
            signals["engagement"] = "high"
        elif time_spent > 20:
            signals["engagement"] = "medium"
        else:
            signals["engagement"] = "low"

        if user_actions.get("copied_text"):
            signals["utility"] = "high"

        if user_actions.get("regenerated"):
            signals["satisfaction"] = "low"

        if user_actions.get("asked_followup"):
            signals["interest"] = "high"

        return signals
