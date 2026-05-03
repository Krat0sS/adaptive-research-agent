# 研究助手 Agent — 架构设计文档

## 1. 项目概述

**项目名称**：自适应研究助手 Agent（Adaptive Research Agent）

**核心能力**：给定一个研究主题，自主搜索文献、提取论点、构建知识图谱、发现研究空白、生成研究提案，并通过与用户的交互持续学习改进——从第一轮的"通用搜索"进化为第五轮的"精准直击"。

## 2. PEAS 精确定义

| 组件 | 定义 |
|------|------|
| **Performance** | `P = α·新颖性(提案) + β·相关性(文献) + γ·结构完整性(报告) + δ·用户隐式满意度(回头率/修改率)`。其中 α/β/γ/δ 通过用户行为动态学习，非静态权重。 |
| **Environment** | 学术信息空间：论文数据库、引用网络、用户历史交互、用户的隐含偏好（部分可观测） |
| **Actuators** | 学术搜索 API 调用、论文摘要提取、知识图谱更新、研究提案生成、检索策略调整 |
| **Sensors** | 用户查询文本、用户对提案的修改/采纳反馈、论文元数据与全文、引用关系、搜索结果质量信号 |

## 3. 环境分类（六维度）

| 维度 | 分类 | 架构影响 |
|------|------|----------|
| 可观测性 | **部分可观测** | 用户偏好不直接暴露，需从行为推断；论文全文可能不可获取 |
| 智能体数量 | **单智能体**（主流程）+ 可选后台评估者 | 简化主流程，评估异步解耦 |
| 确定性 | **随机** | 搜索结果不确定、用户偏好漂移 |
| 片段性 | **序贯式** | 每次交互影响后续策略，用户模型持续累积 |
| 动态性 | **准动态** | 用户偏好缓慢变化，学术领域持续更新 |
| 离散性 | **混合** | 搜索策略离散，相关性/新颖性连续 |

## 4. 核心架构：三层环路

```
┌─────────────────────────────────────────────────────────────┐
│                    元推理层（Meta-Reasoning）                  │
│  职责：决定"继续搜索"还是"生成提案"，控制计算资源分配         │
│  实现：基于边际收益估计的停止规则                              │
│  调用：用户/任务模型查询 → 设定本轮偏好参数                   │
├─────────────────────────────────────────────────────────────┤
│                    慎思层（Deliberation）                     │
│  职责：知识图谱推理、研究空白识别、提案结构化生成              │
│  实现：LLM 驱动的推理 + 图谱上的符号化查询                    │
│  输入：已收集的论文数据 + 用户模型参数                        │
│  输出：结构化研究提案 + 图谱可视化                           │
├─────────────────────────────────────────────────────────────┤
│                    反应层（Reaction）                         │
│  职责：API 调用、论文下载、元数据提取、格式化输出              │
│  实现：工具调用管线 + 错误重试 + 结果缓存                     │
│  特点：毫秒级响应，无推理，纯执行                             │
└─────────────────────────────────────────────────────────────┘
```

## 5. 适应性机制：用户/任务模型

这是项目的灵魂。

**存储结构**（`user_model.json`）：
```json
{
  "user_id": "default",
  "preferences": {
    "venue_weights": {"NeurIPS": 1.2, "ICML": 1.1, "AAAI": 0.9},
    "recency_bias": 0.8,
    "depth_vs_breadth": 0.7,
    "preferred_frameworks": ["SUMO", "CityFlow"],
    "citation_threshold": 10
  },
  "interaction_history": [
    {
      "query": "RL for traffic signal control",
      "search_strategy": {...},
      "user_feedback": {"rating": 3, "modifications": "..."},
      "learned_signals": {"too_broad": true, "missed_venue": "ITSC"}
    }
  ],
  "strategy_performance": {
    "keyword_combos": {"rl+traffic+control": {"success": 0.8, "n": 5}},
    "database_weights": {"semantic_scholar": 0.9, "arxiv": 0.7}
  }
}
```

**适应流程**：
1. 新任务开始 → 查询用户模型 → 获取偏好参数
2. 用偏好参数调整搜索策略（权重、过滤器、排序）
3. 任务完成 → 收集用户反馈信号
4. 更新用户模型（策略性能统计 + 偏好推断）
5. 下次任务自动应用更新后的模型

## 6. 搜索策略自适应：多臂赌博机

**不是让 LLM 决定搜什么关键词**，而是用统计方法学习最优搜索策略。

**臂（Arms）定义**：
- 每个臂 = 一个搜索策略配置（关键词组合 × 数据库 × 过滤器）
- 例如：`("reinforcement learning" + "traffic control", semantic_scholar, venue=NeurIPS|ICML)`

**奖励信号**：
- 用户对返回论文的相关性评分（显式或隐式）
- 论文被纳入最终提案的比例

**算法**：Thompson Sampling（汤普森采样）
- 维护每个臂的 Beta(α, β) 分布
- 每次从每个臂的分布中采样，选择采样值最高的臂
- 根据反馈更新对应臂的 α/β
- 自然平衡探索与利用

## 7. 知识图谱结构

```
节点类型：
  - Paper（论文）：title, authors, year, venue, abstract, citations
  - Concept（概念）：从论文中提取的核心概念
  - Method（方法）：论文使用的方法/框架
  - Finding（发现）：论文的关键结论
  - Gap（空白）：系统推断出的研究空白

边类型：
  - USES(Paper, Method)
  - PRODUCES(Paper, Finding)
  - RELATED_TO(Concept, Concept)
  - CONTRADICTS(Finding, Finding)
  - FILLS(Gap, Paper)  ← 如果某论文填补了某个空白
  - LEAVES(Paper, Gap)  ← 如果某论文遗留了某个空白
```

## 8. 元推理停止规则

**核心问题**：什么时候停止搜索，开始生成提案？

**规则**：当继续搜索的**边际信息增益**低于**计算成本**时停止。

```
估计边际信息增益 = 新论文与已有图谱的新颖度 × 相关性概率
计算成本 = API 调用延迟 + LLM 处理时间 + 用户等待成本

当 估计边际信息增益 < 阈值 τ → 停止搜索，进入提案生成
```

τ 根据用户历史耐心度动态调整——如果用户历史上倾向于快速得到结果，τ 降低（更早停止）。

## 9. 模块划分

```
research-agent/
├── main.py                  # 入口 + CLI 交互
├── config.py                # 配置（API keys, 参数默认值）
├── agent/
│   ├── core.py              # Agent 主循环（三层环路）
│   ├── meta_reasoning.py    # 元推理层：停止规则 + 资源分配
│   └── deliberation.py      # 慎思层：图谱推理 + 提案生成
├── tools/
│   ├── search.py            # 学术搜索工具（Semantic Scholar API）
│   ├── paper_parser.py      # 论文元数据提取
│   └── llm_client.py        # LLM API 调用封装
├── knowledge/
│   ├── graph.py             # 知识图谱数据结构
│   ├── extractor.py         # 从论文提取概念/方法/发现
│   └── gap_analyzer.py      # 研究空白推断
├── learning/
│   ├── user_model.py        # 用户/任务模型持久化
│   ├── bandit.py            # 多臂赌博机（搜索策略自适应）
│   └── feedback.py          # 反馈信号收集与处理
├── output/
│   ├── proposal.py          # 研究提案生成器
│   └── report.py            # 报告格式化输出
├── data/
│   ├── user_model.json      # 持久化的用户模型
│   └── cache/               # 搜索结果缓存
├── tests/
│   ├── test_bandit.py
│   ├── test_graph.py
│   └── test_pipeline.py
├── requirements.txt
└── README.md
```

## 10. 技术栈

| 组件 | 技术选型 | 理由 |
|------|----------|------|
| LLM 引擎 | OpenAI API (GPT-4) | 最强推理能力，API 稳定 |
| 学术搜索 | Semantic Scholar API | 免费、结构化、有引用关系 |
| 知识图谱 | NetworkX (内存图) | 轻量、Python 原生、可视化方便 |
| 搜索策略学习 | 自实现 Thompson Sampling | 轻量、可解释、无外部依赖 |
| 用户模型 | JSON 文件 | 简单可靠，无需数据库 |
| CLI 框架 | Rich + Prompt Toolkit | 美观的终端输出，支持交互 |
| 可选 Web | Streamlit | 一键部署 Web demo，极简代码 |

## 11. 分阶段实施计划

### Phase 1（MVP 核心）
- [ ] Agent 主循环（搜索 → 提取 → 简单报告）
- [ ] Semantic Scholar API 集成
- [ ] LLM 提取论文概念/方法/发现
- [ ] 基础 CLI 交互

### Phase 2（知识图谱 + 慎思层）
- [ ] 知识图谱构建与查询
- [ ] 研究空白推断
- [ ] 结构化提案生成
- [ ] 元推理停止规则

### Phase 3（适应性学习）
- [ ] 用户模型持久化
- [ ] Thompson Sampling 搜索策略
- [ ] 反馈信号收集（显式 + 隐式）
- [ ] 第一轮 vs 第五轮对比演示

### Phase 4（打磨 + 交付）
- [ ] Web Demo（Streamlit）
- [ ] 设计文档完善
- [ ] 演示视频录制
- [ ] 单元测试 + README
