"""Thompson Sampling 多臂赌博机 — 搜索策略自适应"""
import random
from typing import Optional


class ThompsonSamplingBandit:
    """
    Thompson Sampling 多臂赌博机。
    每个臂代表一个搜索策略配置（关键词组合 × 过滤器）。
    通过贝叶斯更新自动平衡探索与利用。
    """

    def __init__(self):
        # 每个臂的 Beta(α, β) 分布参数
        self.arms: dict[str, dict] = {}

    def add_arm(self, arm_id: str, alpha: float = 1.0, beta: float = 1.0):
        """添加一个臂（搜索策略）"""
        if arm_id not in self.arms:
            self.arms[arm_id] = {"alpha": alpha, "beta": beta, "pulls": 0, "rewards": 0.0}

    def select_arm(self, arm_ids: list[str] = None) -> Optional[str]:
        """
        选择一个臂：从每个臂的 Beta 分布中采样，选择采样值最高的。
        
        Args:
            arm_ids: 候选臂 ID 列表。None 则从所有臂中选择。
        
        Returns:
            选中的臂 ID
        """
        candidates = arm_ids or list(self.arms.keys())
        if not candidates:
            return None

        # 确保所有候选臂都已注册
        for aid in candidates:
            if aid not in self.arms:
                self.add_arm(aid)

        # Thompson Sampling：从每个臂的后验分布中采样
        samples = {}
        for aid in candidates:
            arm = self.arms[aid]
            samples[aid] = random.betavariate(arm["alpha"], arm["beta"])

        return max(samples, key=samples.get)

    def update(self, arm_id: str, reward: float):
        """
        更新臂的后验分布。
        
        Args:
            arm_id: 臂 ID
            reward: 奖励值，范围 [0, 1]（0 = 完全不相关，1 = 完全相关）
        """
        if arm_id not in self.arms:
            self.add_arm(arm_id)

        arm = self.arms[arm_id]
        arm["pulls"] += 1
        arm["rewards"] += reward

        # Beta 分布更新：成功 → alpha + reward，失败 → beta + (1 - reward)
        arm["alpha"] += reward
        arm["beta"] += (1 - reward)

    def get_stats(self) -> dict:
        """获取所有臂的统计信息"""
        stats = {}
        for aid, arm in self.arms.items():
            stats[aid] = {
                "pulls": arm["pulls"],
                "mean_reward": arm["rewards"] / max(arm["pulls"], 1),
                "alpha": round(arm["alpha"], 2),
                "beta": round(arm["beta"], 2),
                "ucb": arm["alpha"] / (arm["alpha"] + arm["beta"])  # 期望值
            }
        return stats

    def get_best_arm(self) -> Optional[str]:
        """获取当前期望值最高的臂（利用阶段）"""
        if not self.arms:
            return None
        return max(self.arms, key=lambda a: self.arms[a]["alpha"] / (self.arms[a]["alpha"] + self.arms[a]["beta"]))


def generate_arm_id(keywords: list[str], venue: str = None, year_range: str = None) -> str:
    """生成臂 ID（搜索策略的唯一标识）"""
    parts = ["+".join(sorted(keywords[:3]))]
    if venue:
        parts.append(f"v:{venue}")
    if year_range:
        parts.append(f"y:{year_range}")
    return "|".join(parts)
