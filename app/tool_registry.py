"""工具注册表：注册、参数校验、调用执行（对应文档第 8 章）。

设计要点：
- 接口按真实系统标准设计：name / description / parameters(JSON Schema) / 返回结构；
- V1 返回模拟数据（data/mock/*.json），后续对接真实系统只换实现不改编排；
- 工具权限白名单：仅注册表内的工具可被调用，禁止任意代码执行类工具（文档 8.2）；
- 参数缺失 → 抛 MissingParamError，由编排层走澄清话术；
- 工具异常 → 抛 ToolError，由编排层走 fallback 转人工。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

from . import config


class ToolError(Exception):
    """工具执行异常（超时/业务失败）。"""


class MissingParamError(Exception):
    """参数缺失，需要向用户追问澄清。"""


class Tool:
    def __init__(self, name: str, description: str, parameters: dict[str, Any],
                 handler: Callable[..., dict]):
        self.name = name
        self.description = description
        self.parameters = parameters  # JSON Schema
        self.handler = handler

    def run(self, args: dict[str, Any]) -> dict:
        """参数校验 + 执行。"""
        # 必填参数检查
        required = self.parameters.get("required", [])
        missing = [k for k in required if not args.get(k)]
        if missing:
            raise MissingParamError(f"缺少参数: {', '.join(missing)}")
        return self.handler(**args)


class ToolRegistry:
    def __init__(self):
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def list_schemas(self) -> list[dict]:
        """供模型 function calling 使用的工具描述列表。"""
        return [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters,
                },
            }
            for t in self._tools.values()
        ]

    def names(self) -> list[str]:
        return list(self._tools.keys())


def _load_mock(filename: str) -> Any:
    path: Path = config.MOCK_DIR / filename
    if not path.exists():
        raise ToolError(f"模拟数据缺失: {path.name}")
    return json.loads(path.read_text(encoding="utf-8"))


# ---------- 工具实现（V1 模拟） ----------

def _query_order(order_id: str) -> dict:
    """查询订单状态。"""
    orders = _load_mock("orders.json")
    order = next((o for o in orders if o["order_id"] == order_id.strip()), None)
    if not order:
        raise ToolError(f"未找到订单 {order_id}，请核对订单号")
    return {"found": True, "order": order}


def _query_logistics(order_id: str) -> dict:
    """查询物流轨迹。"""
    orders = _load_mock("orders.json")
    logistics = _load_mock("logistics.json")
    order = next((o for o in orders if o["order_id"] == order_id.strip()), None)
    if not order:
        raise ToolError(f"未找到订单 {order_id}，请核对订单号")
    if not order.get("logistics_no"):
        return {"found": True, "order_id": order_id, "note": "该订单尚未发货，暂无物流信息"}
    traces = logistics.get(order["logistics_no"], [])
    if not traces:
        return {"found": True, "order_id": order_id, "note": "物流信息同步中，请稍后查询"}
    return {
        "found": True,
        "order_id": order_id,
        "logistics_no": order["logistics_no"],
        "carrier": "顺丰速运" if order["logistics_no"].startswith("SF") else "中通快递",
        "latest": traces[-1],
        "traces": traces,
    }


def _transfer_human(session_id: str, summary: str) -> dict:
    """转人工并携带交接包（V1 记录交接包，人工侧留接口）。"""
    return {
        "transferred": True,
        "session_id": session_id,
        "summary": summary,
        "note": "已记录交接信息包，人工客服侧对接预留",
    }


def build_registry() -> ToolRegistry:
    reg = ToolRegistry()
    reg.register(Tool(
        name="query_order",
        description="查询订单状态与基本信息（需完整订单号，格式如 O20260824001）",
        parameters={
            "type": "object",
            "properties": {
                "order_id": {"type": "string",
                             "description": "完整订单号，形如 O20260824001"},
            },
            "required": ["order_id"],
        },
        handler=_query_order,
    ))
    reg.register(Tool(
        name="query_logistics",
        description="查询物流轨迹与最新状态（需完整订单号）",
        parameters={
            "type": "object",
            "properties": {
                "order_id": {"type": "string",
                             "description": "完整订单号，形如 O20260824001"},
            },
            "required": ["order_id"],
        },
        handler=_query_logistics,
    ))
    reg.register(Tool(
        name="transfer_human",
        description="转接人工客服，携带会话摘要（用户投诉、明确要求人工或 AI 无法解决时调用）",
        parameters={
            "type": "object",
            "properties": {
                "session_id": {"type": "string", "description": "当前会话 ID"},
                "summary": {"type": "string", "description": "会话摘要，供人工客服快速了解"},
            },
            "required": ["session_id", "summary"],
        },
        handler=_transfer_human,
    ))
    return reg
