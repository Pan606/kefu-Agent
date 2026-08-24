"""意图路由：基于模型的结构化分类 + 规则兜底（对应文档 4.2 intent_router）。

设计要点：
- 主路径用模型结构化输出（Pydantic）分类六类意图；
- 规则兜底：命中订单号格式 / "物流"等关键词时直接判定，模型失败时可降级；
- 分类结果写入 State.intent，供 LangGraph 条件边分流。
"""
from __future__ import annotations

import json
import re

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from . import config
from . import prompts

# 订单号格式：O + 11 位数字（与模拟数据一致）
ORDER_ID_PATTERN = re.compile(r"O\d{11}")

# 关键词规则兜底
KEYWORD_RULES: list[tuple[re.Pattern, str]] = [
    (re.compile(r"(订单号|订单状态|我的订单|查询订单|订单信息)"), "order"),
    (re.compile(r"(物流|快递|派送|发货了没有|到哪了)"), "logistics"),
    (re.compile(r"(转人工|人工客服|投诉|找人工|人工服务)"), "transfer"),
    (re.compile(r"(你好|您好|在吗|谢谢|再见|拜拜|哈哈|嗯)"), "chitchat"),
]


class IntentResult(BaseModel):
    intent: str = Field(description="knowledge|order|logistics|chitchat|forbidden|transfer")
    reason: str = Field(description="判断理由")


def classify_intent(llm, question: str, history: list[dict]) -> tuple[str, str]:
    """意图分类。返回 (intent, reason)。

    顺序：规则兜底（确定性）→ 模型分类（灵活性）。
    """
    # 1) 规则兜底：订单号格式直接判定
    if ORDER_ID_PATTERN.search(question):
        if "物流" in question or "快递" in question:
            return "logistics", "规则：消息含订单号且涉及物流关键词"
        return "order", "规则：消息含完整订单号"

    # 2) 关键词兜底
    for pat, intent in KEYWORD_RULES:
        if pat.search(question):
            return intent, f"规则：命中关键词 {pat.pattern}"

    # 3) 模型分类
    try:
        history_text = "\n".join(
            f"{'用户' if m['role'] == 'user' else '客服'}: {m['content']}"
            for m in history[-4:])
        structured_llm = llm.with_structured_output(IntentResult)
        result = structured_llm.invoke([
            SystemMessage(content=prompts.INTENT_SYSTEM_PROMPT),
            HumanMessage(
                content=f"历史对话：\n{history_text or '（无）'}\n\n当前用户消息：{question}"),
        ])
        return result.intent, result.reason
    except Exception:
        # 结构化解析失败（如模型输出双花括号/代码块）→ 手动容错解析
        try:
            raw = llm.invoke([
                SystemMessage(content=prompts.INTENT_SYSTEM_PROMPT),
                HumanMessage(
                    content=f"历史对话：\n{history_text or '（无）'}\n\n当前用户消息：{question}"),
            ]).content
            result = _parse_intent_json(str(raw))
            return result.intent, result.reason
        except Exception as e:
            # 模型分类失败降级为 knowledge，让下游兜底
            return "knowledge", f"模型分类失败，降级为 knowledge（{e.__class__.__name__}）"


def _parse_intent_json(raw: str) -> IntentResult:
    """容错解析模型输出的 JSON：兼容代码块 / 双花括号 / 前后杂文本。"""
    text = raw.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.M).strip()
    if text.startswith("{{") and text.endswith("}}"):
        text = text[1:-1]
    start, end = text.find("{"), text.rfind("}")
    if start >= 0 and end > start:
        text = text[start:end + 1]
    data = json.loads(text)
    return IntentResult(**data)


def route_to_qa(intent: str) -> bool:
    return intent in ("knowledge",)


def route_to_tool(intent: str) -> bool:
    return intent in ("order", "logistics")
