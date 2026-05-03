"""配置文件 — API keys 和默认参数"""
import os

# LLM 配置
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")  # 默认用 mini 控制成本
OPENAI_MAX_TOKENS = 4000

# Semantic Scholar 配置
S2_API_KEY = os.environ.get("S2_API_KEY", "")  # 可选，有 key 速率更高
S2_BASE_URL = "https://api.semanticscholar.org/graph/v1"
S2_RATE_LIMIT_DELAY = 1.0  # 无 key 时每秒最多 1 次请求

# Agent 参数
MAX_PAPERS_PER_SEARCH = 20      # 每次搜索最多获取论文数
MAX_SEARCH_ROUNDS = 5           # 最大搜索轮数
MIN_PAPERS_FOR_ANALYSIS = 5     # 进入分析阶段的最少论文数
META_REASONING_THRESHOLD = 0.3  # 元推理停止阈值（边际信息增益 < 此值时停止）

# 用户模型
USER_MODEL_PATH = os.path.join(os.path.dirname(__file__), "data", "user_model.json")

# 知识图谱
GRAPH_CACHE_PATH = os.path.join(os.path.dirname(__file__), "data", "graph_cache.json")
