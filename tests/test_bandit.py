"""单元测试 — 多臂赌博机"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from learning.bandit import ThompsonSamplingBandit, generate_arm_id


def test_basic_selection():
    """测试基本选择功能"""
    bandit = ThompsonSamplingBandit()
    bandit.add_arm("arm_a")
    bandit.add_arm("arm_b")

    selected = bandit.select_arm()
    assert selected in ["arm_a", "arm_b"], f"Unexpected selection: {selected}"
    print("✓ test_basic_selection passed")


def test_learning():
    """测试学习效果：持续奖励一个臂，它应该被更频繁地选择"""
    bandit = ThompsonSamplingBandit()
    bandit.add_arm("good_arm")
    bandit.add_arm("bad_arm")

    # 模拟 100 轮：good_arm 总是得 1.0，bad_arm 总是得 0.0
    for _ in range(100):
        bandit.update("good_arm", 1.0)
        bandit.update("bad_arm", 0.0)

    # good_arm 的期望值应该远高于 bad_arm
    stats = bandit.get_stats()
    assert stats["good_arm"]["mean_reward"] > 0.9
    assert stats["bad_arm"]["mean_reward"] < 0.1

    # 最优臂应该是 good_arm
    best = bandit.get_best_arm()
    assert best == "good_arm", f"Expected good_arm, got {best}"
    print("✓ test_learning passed")


def test_exploration():
    """测试探索行为：即使有一个臂明显更好，也应该偶尔探索其他臂"""
    bandit = ThompsonSamplingBandit()
    bandit.add_arm("best_arm")
    bandit.add_arm("other_arm")

    # 微弱强化 best_arm（让两个臂的分布仍有重叠，保留探索空间）
    for _ in range(3):
        bandit.update("best_arm", 0.8)
        bandit.update("other_arm", 0.4)

    # 选择 1000 次，统计选择分布
    selections = {"best_arm": 0, "other_arm": 0}
    for _ in range(1000):
        selected = bandit.select_arm()
        selections[selected] += 1

    # best_arm 应该被选择更多次，但 other_arm 也应该被探索（分布有重叠）
    assert selections["best_arm"] > 400, f"best_arm selected too few: {selections['best_arm']}"
    assert selections["other_arm"] > 10, f"other_arm explored too few: {selections['other_arm']}"
    print(f"✓ test_exploration passed (best: {selections['best_arm']}, other: {selections['other_arm']})")


def test_arm_id_generation():
    """测试臂 ID 生成"""
    arm_id = generate_arm_id(["rl", "traffic"], venue="NeurIPS", year_range="2020-2025")
    assert "rl" in arm_id
    assert "traffic" in arm_id
    assert "NeurIPS" in arm_id
    print(f"✓ test_arm_id_generation passed: {arm_id}")


def test_empty_bandit():
    """测试空赌博机"""
    bandit = ThompsonSamplingBandit()
    assert bandit.select_arm() is None
    assert bandit.get_best_arm() is None
    print("✓ test_empty_bandit passed")


if __name__ == "__main__":
    test_basic_selection()
    test_learning()
    test_exploration()
    test_arm_id_generation()
    test_empty_bandit()
    print("\n所有赌博机测试通过 ✓")
