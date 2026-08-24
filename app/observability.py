"""可观测性模块：Langfuse 追踪 + 结构化日志（对应文档第 10 章）。

设计要点：
- 无 Langfuse key 时自动降级为 noop（本地开发不阻塞）；
- 每次对话打 trace，记录输入/输出/检索/工具/耗时；
- 隐私合规（文档 9.3）：用户消息与画像不写入日志明文，仅记录元数据。
"""
from __future__ import annotations

import logging
import time
from typing import Any

from . import config

logger = logging.getLogger("kefu_agent")


class NoopTrace:
    """无 Langfuse 时的降级实现。"""

    def __init__(self, *args, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def update(self, **kwargs):
        pass

    def generation(self, *args, **kwargs):
        return self

    def span(self, *args, **kwargs):
        return self


_trace_cls: type = NoopTrace


def init_observability() -> None:
    """初始化 Langfuse（可选）。失败时保持 noop。"""
    global _trace_cls
    if not config.LANGFUSE_ENABLED:
        logger.info("Langfuse 未配置（LANGFUSE_PUBLIC_KEY/SECRET_KEY 为空），使用 noop 追踪")
        return
    try:
        from langfuse.callback import CallbackHandler  # noqa: F401
        from langfuse.decorators import langfuse_context, observe  # noqa: F401

        class LangfuseTrace:
            """基于 langfuse_context 的对话级 trace。"""

            def __init__(self, name: str, trace_id: str, metadata: dict | None = None):
                self.name = name
                self.trace_id = trace_id
                self.metadata = metadata or {}

            def __enter__(self):
                langfuse_context.update_current_trace(
                    name=self.name, trace_id=self.trace_id,
                    metadata=self.metadata, tags=["v0.1.0"])
                return self

            def __exit__(self, *args):
                return False

            def update(self, **kwargs):
                langfuse_context.update_current_trace(**kwargs)

        _trace_cls = LangfuseTrace
        logger.info("Langfuse 已启用（host=%s）", config.LANGFUSE_HOST)
    except ImportError:
        logger.warning("未安装 langfuse 包，使用 noop 追踪")


def trace(name: str, trace_id: str, metadata: dict | None = None):
    """创建一次对话追踪（noop 或 Langfuse）。"""
    return _trace_cls(name=name, trace_id=trace_id, metadata=metadata)


class Metrics:
    """核心指标计数器（文档 10.2）：命中率 / 兜底率 / 转人工率 / 响应耗时。"""

    def __init__(self):
        self.reset()

    def reset(self) -> None:
        self.total = 0
        self.hits = 0          # 知识库命中
        self.fallback = 0      # 走澄清/转人工
        self.transfer = 0      # 转人工
        self.reject = 0        # 违禁拒绝
        self.response_ms: list[float] = []

    def record(self, intent: str, used_rag: bool, fallback_type: str,
               transfer: bool, reject: bool, elapsed_ms: float) -> None:
        self.total += 1
        if used_rag:
            self.hits += 1
        if fallback_type:
            self.fallback += 1
        if transfer:
            self.transfer += 1
        if reject:
            self.reject += 1
        self.response_ms.append(elapsed_ms)

    def snapshot(self) -> dict[str, Any]:
        n = max(self.total, 1)
        p95 = sorted(self.response_ms)[
            int(len(self.response_ms) * 0.95) - 1] if self.response_ms else 0
        return {
            "total": self.total,
            "kb_hit_rate": round(self.hits / n, 3),
            "fallback_rate": round(self.fallback / n, 3),
            "transfer_rate": round(self.transfer / n, 3),
            "reject_count": self.reject,
            "avg_response_ms": round(sum(self.response_ms) / n, 1) if self.response_ms else 0,
            "p95_response_ms": round(p95, 1),
        }


metrics = Metrics()
