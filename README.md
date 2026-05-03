# 自适应研究助手 Agent（Adaptive Research Agent）

> 毕业项目 — 基于 AI Agent 架构的自适应学术研究助手

## 项目简介

给定一个研究主题，Agent 能够：

1. **自主搜索**学术文献（Semantic Scholar API）
2. **提取**论文核心概念、方法和发现
3. **构建**知识图谱（NetworkX）
4. **识别**研究空白
5. **生成**结构化研究提案
6. **学习**：通过用户反馈持续改进搜索策略（Thompson Sampling）

### 核心特性

- **三层架构**：反应层（搜索执行）→ 慎思层（图谱推理）→ 元推理层（资源分配）
- **自适应搜索**：多臂赌博机自动学习最优关键词组合和数据库策略
- **用户偏好建模**：从交互历史中推断用户偏好，个性化搜索结果
- **知识图谱可视化**：文本格式的图谱摘要，展示论文-概念-空白关系

## 快速开始

### 环境要求

- Python 3.10+
- OpenAI API Key

### 安装

```bash
cd research-agent
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 配置

```bash
export OPENAI_API_KEY="your-api-key-here"
export OPENAI_MODEL="gpt-4o-mini"  # 可选，默认 gpt-4o-mini
```

### 运行

```bash
python main.py
```

输入研究主题，等待 Agent 搜索、分析、生成提案。完成后可以给出反馈，Agent 会学习你的偏好。

## 架构概览

```
┌─────────────────────────────────────────┐
│          元推理层 (Meta-Reasoning)        │
│  • 停止规则：边际信息增益 < 阈值时停止    │
│  • 策略选择：Thompson Sampling            │
│  • 偏好查询：用户模型持久化              │
├─────────────────────────────────────────┤
│          慎思层 (Deliberation)            │
│  • 知识图谱推理                          │
│  • 研究空白识别（LLM 驱动）              │
│  • 提案生成（结构化 JSON）               │
├─────────────────────────────────────────┤
│          反应层 (Reaction)                │
│  • Semantic Scholar API 调用             │
│  • 论文元数据提取                        │
│  • 概念/方法/发现提取                    │
└─────────────────────────────────────────┘
```

## 项目结构

```
research-agent/
├── main.py                  # CLI 入口
├── config.py                # 配置
├── agent/
│   └── core.py              # Agent 主循环
├── tools/
│   ├── llm_client.py        # OpenAI API 封装
│   ├── search.py            # Semantic Scholar 搜索
│   └── paper_parser.py      # 论文解析
├── knowledge/
│   ├── graph.py             # 知识图谱
│   ├── extractor.py         # 知识提取
│   └── gap_analyzer.py      # 研究空白分析
├── learning/
│   ├── user_model.py        # 用户偏好模型
│   ├── bandit.py            # Thompson Sampling
│   └── feedback.py          # 反馈收集
├── output/
│   └── proposal.py          # 提案生成
├── data/                    # 持久化数据
├── tests/                   # 单元测试
├── DESIGN.md                # 详细设计文档
└── README.md
```

## 使用示例

```
📝 输入研究主题: reinforcement learning for traffic signal control

══════════════════════════════════════════════════
🔬 研究主题: reinforcement learning for traffic signal control
📋 任务 ID: a3f2b1c8
👤 历史交互: 0 次
══════════════════════════════════════════════════

── 搜索轮次 1/5 ──
  策略: keywords=['reinforcement', 'learning', 'traffic'], venue=不限
  获取: 20 篇论文
  提取: 8 个概念, 5 个方法

── 搜索轮次 2/5 ──
  策略: keywords=['reinforcement', 'learning', 'signal'], venue=NeurIPS
  获取: 15 篇论文
  提取: 3 个概念, 2 个方法

  ✅ 元推理判定：边际信息增益不足，停止搜索

── 慎思层：分析研究空白 ──
  ⚠ 多智能体协调在真实交通网络中的可扩展性问题
  ⚠ 从模拟到真实环境的迁移差距

── 慎思层：生成研究提案 ──

📄 研究提案
══════════════════════════════════════════════════
📌 标题: Scalable Multi-Agent Reinforcement Learning...
...
```

## 设计文档

详见 [DESIGN.md](DESIGN.md)

## License

MIT
