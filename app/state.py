"""LangGraph 状态定义（对应文档 5.1 状态核心字段）。"""
from __future__ import annotations

from typing import Annotated, TypedDict

from langgraph.graph.message import add_messages


class AgentState(TypedDict, total=False):
    # --- 会话基础 ---
    session_id: str

    # --- 输入 ---
    user_input: str            # 原始用户消息
    filtered_input: str        # 安全过滤后的消息

    # --- 对话历史（LangChain 消息，add_messages 自动追加） ---
    messages: Annotated[list, add_messages]

    # --- 意图路由 ---
    intent: str                # knowledge/order/logistics/chitchat/forbidden/transfer
    intent_reason: str

    # --- 能力层结果 ---
    retrieved_docs: list       # [{content, score, source, section}]
    tool_results: dict         # 工具返回
    citations: list            # 来源引用

    # --- 兜底 ---
    fallback_type: str         # "" / clarify / transfer
    clarify_round: int         # 已澄清轮数
    reject_reason: str         # 违禁原因类别

    # --- 输出 ---
    response: str
    transfer_package: dict     # 交接信息包
    used_fallback_model: bool  # 是否降级到备选模型
