"""用户/任务模型 — 持久化学习用户偏好"""
import json
import os
from datetime import datetime
import config


DEFAULT_MODEL = {
    "user_id": "default",
    "preferences": {
        "venue_weights": {},
        "recency_bias": 0.7,
        "depth_vs_breadth": 0.5,
        "preferred_frameworks": [],
        "citation_threshold": 5
    },
    "interaction_history": [],
    "strategy_performance": {
        "keyword_combos": {},
        "database_weights": {"semantic_scholar": 1.0}
    }
}


class UserModel:
    """用户偏好模型，支持持久化和在线更新"""

    def __init__(self, path: str = None):
        self.path = path or config.USER_MODEL_PATH
        self.data = self._load()

    def _load(self) -> dict:
        if os.path.exists(self.path):
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                pass
        return DEFAULT_MODEL.copy()

    def save(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self.data, f, ensure_ascii=False, indent=2)

    def get_preferences(self) -> dict:
        """获取当前偏好参数"""
        return self.data.get("preferences", DEFAULT_MODEL["preferences"])

    def record_interaction(self, query: str, strategy: dict, feedback: dict):
        """记录一次交互（用于后续学习）"""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "strategy": strategy,
            "feedback": feedback
        }
        self.data.setdefault("interaction_history", []).append(entry)

        # 保留最近 50 条
        self.data["interaction_history"] = self.data["interaction_history"][-50:]

        # 根据反馈更新偏好
        self._update_from_feedback(feedback, strategy)
        self.save()

    def _update_from_feedback(self, feedback: dict, strategy: dict):
        """
        根据用户反馈更新偏好。
        
        改进点：
        1. 增加 rating <= 2 的对称惩罚，防止权重只增不减
        2. 每次更新时对 venue_weights 施加微小衰减，防止长期膨胀
        """
        prefs = self.data["preferences"]

        # 如果用户给了评分
        rating = feedback.get("rating")
        if rating is not None:
            # 如果用户修改了提案，说明偏好有偏差
            if feedback.get("modifications"):
                prefs["depth_vs_breadth"] = max(0.1, prefs["depth_vs_breadth"] - 0.05)

            venue = strategy.get("venue")
            if venue:
                weights = prefs.get("venue_weights", {})

                # 全局微衰减：每次更新时所有权重乘以 0.99，防止无限膨胀
                for k in weights:
                    weights[k] *= 0.99

                if rating >= 4:
                    # 高评分 = 当前策略有效，强化偏好
                    weights[venue] = weights.get(venue, 1.0) + 0.1
                elif rating <= 2:
                    # 低评分 = 当前策略不佳，惩罚对应 venue
                    weights[venue] = max(0.1, weights.get(venue, 1.0) - 0.1)

                prefs["venue_weights"] = weights

            # 更新 recency_bias：高评分倾向于保持当前偏好，低评分倾向于探索
            if rating >= 4:
                prefs["recency_bias"] = min(1.0, prefs.get("recency_bias", 0.7) + 0.02)
            elif rating <= 2:
                prefs["recency_bias"] = max(0.1, prefs.get("recency_bias", 0.7) - 0.05)

        # 如果用户反馈论文太少/太多
        if feedback.get("too_few_papers"):
            prefs["citation_threshold"] = max(0, prefs["citation_threshold"] - 2)
        if feedback.get("too_many_papers"):
            prefs["citation_threshold"] = prefs.get("citation_threshold", 5) + 2

    def get_venue_weights(self) -> dict:
        """获取会议/期刊偏好权重"""
        return self.data.get("preferences", {}).get("venue_weights", {})

    def get_history_summary(self) -> str:
        """获取历史交互摘要（供 LLM 参考）"""
        history = self.data.get("interaction_history", [])
        if not history:
            return "无历史交互记录。"

        recent = history[-5:]
        lines = []
        for h in recent:
            fb = h.get("feedback", {})
            lines.append(f"- 查询: {h['query']}")
            if fb.get("rating"):
                lines.append(f"  评分: {fb['rating']}/5")
            if fb.get("modifications"):
                lines.append(f"  用户修改: {fb['modifications'][:100]}")

        return "\n".join(lines)
