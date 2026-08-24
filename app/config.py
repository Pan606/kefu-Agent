"""配置模块：加载 .env 并提供模型抽象层（主/备模型一键切换）。

设计要点（对应设计文档 3.2 模型选型说明）：
- glm-4-flash 为免费档，存在限流与偶发不稳定风险，必须预留模型抽象层；
- 统一 get_llm() 接口，provider 支持 zhipu / deepseek，主模型异常时切 fallback；
- 对话模型与 embedding 模型分离配置；
- 检索参数（Top-K / 阈值）不写死在代码里，可经 .env 调整。
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env", override=False)


# ---------- 路径 ----------
DATA_DIR = PROJECT_ROOT / "data"
KB_DIR = DATA_DIR / "kb"
MOCK_DIR = DATA_DIR / "mock"
STORE_DIR = DATA_DIR / "store"
CHROMA_DIR = STORE_DIR / "chroma"
CHROMA_COLLECTION = "kefu_kb"  # Chroma 要求 collection 名 3-512 字符
DB_PATH = STORE_DIR / "kefu.db"
WEB_DIR = PROJECT_ROOT / "web"

# ---------- 模型 ----------
ZHIPU_BASE_URL = os.getenv("ZHIPU_BASE_URL", "https://open.bigmodel.cn/api/paas/v4/")
ZHIPU_API_KEY = os.getenv("ZHIPU_API_KEY", "")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "zhipu")
LLM_MODEL = os.getenv("LLM_MODEL", "glm-4-flash")
LLM_FALLBACK_PROVIDER = os.getenv("LLM_FALLBACK_PROVIDER", "deepseek")
LLM_FALLBACK_MODEL = os.getenv("LLM_FALLBACK_MODEL", "deepseek-v4-flash")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "embedding-3")

# ---------- 检索参数 ----------
# [M1 调参结论] 智谱 embedding-3 的相关分数整体偏低（正确文档约 0.35-0.65），
# 文档草案的 0.5 阈值过高导致召回不足，实测调到 0.3；可在 .env 覆盖。
RETRIEVAL_TOP_K = int(os.getenv("RETRIEVAL_TOP_K", "4"))
RETRIEVAL_THRESHOLD = float(os.getenv("RETRIEVAL_THRESHOLD", "0.3"))

# ---------- 对话窗口 ----------
MAX_CONTEXT_TOKENS = int(os.getenv("MAX_CONTEXT_TOKENS", "8000"))
MAX_HISTORY_ROUNDS = int(os.getenv("MAX_HISTORY_ROUNDS", "10"))

# ---------- 可观测性 ----------
LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY", "")
LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY", "")
LANGFUSE_HOST = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
LANGFUSE_ENABLED = bool(LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY)


def get_llm(provider: str | None = None, model: str | None = None,
            temperature: float = 0.3, use_fallback: bool = False):
    """模型抽象层：按 provider 构造 OpenAI 兼容的 ChatOpenAI。

    Args:
        provider: zhipu / deepseek；None 时按 use_fallback 取主/备。
        model: 模型名；None 时按 provider 取默认。
        temperature: 采样温度。
        use_fallback: True 时强制使用备选模型（主模型异常时降级）。
    """
    from langchain_openai import ChatOpenAI

    if provider is None:
        provider = LLM_FALLBACK_PROVIDER if use_fallback else LLM_PROVIDER
    if model is None:
        model = LLM_FALLBACK_MODEL if use_fallback else LLM_MODEL

    if provider == "zhipu":
        api_key, base_url = ZHIPU_API_KEY, ZHIPU_BASE_URL
    elif provider == "deepseek":
        api_key, base_url = DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL
    else:
        raise ValueError(f"未知 provider: {provider}（支持 zhipu / deepseek）")

    if not api_key:
        raise RuntimeError(
            f"未找到 {provider.upper()}_API_KEY：请在项目根目录 .env 中配置")

    return ChatOpenAI(
        model=model,
        api_key=api_key,
        base_url=base_url,
        temperature=temperature,
        timeout=60,
        max_retries=1,
    )


def get_embeddings():
    """Embedding 模型：智谱 embedding-3（2048 维，OpenAI 兼容）。"""
    from langchain_openai import OpenAIEmbeddings

    if not ZHIPU_API_KEY:
        raise RuntimeError("未找到 ZHIPU_API_KEY：请在项目根目录 .env 中配置")
    return OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        api_key=ZHIPU_API_KEY,
        base_url=ZHIPU_BASE_URL,
    )
