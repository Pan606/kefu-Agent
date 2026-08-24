"""LangGraph 状态图：安全 → 意图路由 → 检索/工具/拒绝 → 生成 → 兜底 → 转人工。

对应设计文档 5.1 状态图设计与 5.2 兜底策略（L1 澄清 / L2 转人工 / L3 拒绝）。

节点：
  safety_check   → 输入安全过滤（敏感词/注入），拦截走 reject
  intent_router  → 意图分类，条件边分流
  rag_retrieve   → 知识库检索，无命中走 fallback（L1）
  tool_execute   → 工具调用（query_order/query_logistics），失败走 fallback（L2）
  generate       → 组装上下文，模型生成回复
  reject         → 违禁委婉拒绝（L3）
  fallback       → 澄清或转人工决策（最多 2 轮澄清）
  transfer_human → 生成交接信息包，记录到 SQLite
"""
from __future__ import annotations

import json
import time

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END, START, StateGraph

from . import config, prompts, rag_engine, safety
from . import intent_router as ir
from . import memory
from . import observability
from . import tool_registry as tr
from .observability import logger
from .state import AgentState

MAX_CLARIFY_ROUNDS = 2


# ---------- 节点实现 ----------

def safety_check(state: AgentState) -> dict:
    """输入安全过滤。命中敏感词/注入 → fallback_type=reject。"""
    result = safety.check_input(state["user_input"])
    if result.blocked:
        logger.info("safety: 拦截 %s (matched=%s)", result.category, result.matched)
        return {
            "filtered_input": state["user_input"],
            "fallback_type": "reject",
            "reject_reason": result.category,
        }
    return {"filtered_input": result.text}


def intent_router(state: AgentState) -> dict:
    """意图分类。"""
    llm = _get_llm(state)
    history = _history_dicts(state)
    intent, reason = ir.classify_intent(
        llm, state["filtered_input"], history)
    logger.info("intent: %s (%s)", intent, reason)
    return {"intent": intent, "intent_reason": reason}


def rag_retrieve(state: AgentState) -> dict:
    """知识库检索 + 阈值过滤。"""
    docs = rag_engine.retrieve(state["filtered_input"])
    logger.info("rag: 检索到 %d 条（阈值 %s）", len(docs), config.RETRIEVAL_THRESHOLD)
    if not docs:
        return {"retrieved_docs": [], "fallback_type": "clarify"}
    return {"retrieved_docs": docs}


def tool_execute(state: AgentState) -> dict:
    """工具调用。参数缺失→澄清；执行失败→转人工。"""
    registry = tr.build_registry()
    tool_name = "query_order" if state["intent"] == "order" else "query_logistics"
    tool = registry.get(tool_name)
    try:
        # 从消息中提取订单号
        order_id = ir.ORDER_ID_PATTERN.search(state["filtered_input"])
        args = {"order_id": order_id.group(0)} if order_id else {}
        result = tool.run(args)
        logger.info("tool %s 成功", tool_name)
        return {"tool_results": {"tool": tool_name, "data": result}}
    except tr.MissingParamError as e:
        logger.info("tool %s 参数缺失: %s", tool_name, e)
        return {"fallback_type": "clarify",
                "clarify_hint": "请提供您的订单号（格式：O 开头加 11 位数字）"}
    except tr.ToolError as e:
        logger.info("tool %s 失败: %s", tool_name, e)
        return {"fallback_type": "transfer", "tool_error": str(e)}
    except Exception as e:
        logger.exception("tool %s 异常", tool_name)
        return {"fallback_type": "transfer", "tool_error": str(e)}


def generate(state: AgentState) -> dict:
    """组装上下文并生成回复。模型异常自动降级到备选模型。"""
    llm = _get_llm(state)
    history = _history_dicts(state)

    # 组装上下文
    context_parts = []
    if state.get("retrieved_docs"):
        context_parts.append(
            "【知识库检索结果】\n" + rag_engine.format_context(state["retrieved_docs"]))
    if state.get("tool_results"):
        context_parts.append(
            "【工具返回结果】\n" + json.dumps(state["tool_results"], ensure_ascii=False,
                                             indent=1))

    history_text = "\n".join(
        f"{'用户' if m['role'] == 'user' else '客服'}: {m['content']}"
        for m in history[-config.MAX_HISTORY_ROUNDS * 2:])

    prompt = prompts.SYSTEM_PROMPT.format(
        context=context_parts[0] if context_parts else "（本次对话无需检索，请直接礼貌作答）",
        history=history_text or "（无）",
        question=state["filtered_input"],
    )

    try:
        reply = llm.invoke([HumanMessage(content=prompt)]).content
    except Exception as e:
        # 主模型失败 → 降级备选模型（文档 3.2 / 风险对策）
        logger.warning("主模型调用失败（%s），尝试备选模型", e.__class__.__name__)
        try:
            fallback_llm = _get_llm(state, use_fallback=True)
            reply = fallback_llm.invoke([HumanMessage(content=prompt)]).content
            return {"response": str(reply), "used_fallback_model": True}
        except Exception as e2:
            logger.error("备选模型也失败: %s", e2)
            return {"fallback_type": "transfer", "tool_error": "模型服务不可用"}

    reply = safety.desensitize(str(reply))

    # 附上引用（检索命中时）
    citations = []
    if state.get("retrieved_docs"):
        citations = rag_engine.format_citations(state["retrieved_docs"])
        if citations:
            reply += "\n\n【来源】" + "；".join(citations)

    return {"response": reply, "citations": citations}


def reject(state: AgentState) -> dict:
    """违禁内容委婉拒绝（L3）。"""
    llm = _get_llm(state)
    reason = state.get("reject_reason", "违规")
    try:
        resp = llm.invoke([HumanMessage(content=prompts.REJECT_PROMPT.format(
            reason=reason, question=state["filtered_input"]))]).content
        reply = safety.desensitize(str(resp))
    except Exception:
        reply = "抱歉，您的问题涉及敏感或不适宜的内容，我无法回答。如有其他问题，欢迎继续咨询。"
    return {"response": reply, "fallback_type": ""}


def fallback(state: AgentState) -> dict:
    """兜底决策：L1 澄清 / L2 转人工。"""
    ftype = state.get("fallback_type", "clarify")
    rounds = state.get("clarify_round", 0)

    if ftype == "clarify" and rounds < MAX_CLARIFY_ROUNDS:
        # L1：追问澄清
        hint = state.get("clarify_hint")
        if hint:
            reply = hint
        elif state.get("intent") == "order":
            reply = "请问您要查询的是订单状态吗？如果是，请提供您的订单号（O 开头加 11 位数字）。"
        elif state.get("intent") == "logistics":
            reply = "请问您要查询的是物流信息吗？如果是，请提供您的订单号（O 开头加 11 位数字）。"
        else:
            reply = ("抱歉，我没有完全理解您的意思。您是想咨询产品信息、价格、售后、"
                     "物流还是政策方面的问题呢？请具体描述一下。")
        return {"response": reply, "fallback_type": "", "clarify_round": rounds + 1}

    # L2：转人工
    return {"fallback_type": "transfer"}


def transfer_human(state: AgentState) -> dict:
    """转人工：生成交接信息包并记录（对应文档 5.3）。"""
    session_id = state["session_id"]
    history = _history_dicts(state)

    # 会话摘要（最近 5 轮）
    try:
        llm = _get_llm(state)
        summary = llm.invoke([HumanMessage(content=prompts.TRANSFER_SUMMARY_PROMPT.format(
            history=json.dumps(history[-10:], ensure_ascii=False)))])
        summary_text = str(summary.content)[:200]
    except Exception:
        summary_text = "（摘要生成失败）"

    profile = memory.get_profile(session_id)
    handoff = {
        "session_id": session_id,
        "summary": summary_text,
        "profile": {
            "preferences": profile["preferences"],
            "topics": profile["topics"],
        },
        "intent": state.get("intent", ""),
        "tool_results": state.get("tool_results"),
        "fallback_reason": state.get("tool_error", state.get("intent_reason", "")),
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    memory.save_transfer(session_id, handoff)
    return {
        "transfer_package": handoff,
        "response": prompts.FALLBACK_TRANSFER,
        "fallback_type": "",
    }


# ---------- 条件边路由 ----------

def _route_after_safety(state: AgentState) -> str:
    if state.get("fallback_type") == "reject":
        return "reject"
    return "intent_router"


def _route_after_intent(state: AgentState) -> str:
    intent = state.get("intent", "")
    if intent == "forbidden":
        return "reject"
    if ir.route_to_tool(intent):
        return "tool_execute"
    if ir.route_to_qa(intent):
        return "rag_retrieve"
    if intent == "transfer":
        return "transfer_human"
    return "generate"  # chitchat 等直接生成


def _route_after_rag(state: AgentState) -> str:
    if state.get("retrieved_docs"):
        return "generate"
    return "fallback"


def _route_after_tool(state: AgentState) -> str:
    if state.get("tool_results") is not None:
        return "generate"
    return "fallback"


def _route_after_fallback(state: AgentState) -> str:
    if state.get("fallback_type") == "transfer":
        return "transfer_human"
    return END


# ---------- 辅助 ----------

def _get_llm(state: AgentState, use_fallback: bool = False):
    """按状态取模型；模型实例缓存于图的注入，避免重复构造。"""
    # 通过 module-level 缓存降低重复构造开销
    if use_fallback:
        return _FALLBACK_LLM
    return _PRIMARY_LLM


def _history_dicts(state: AgentState) -> list[dict]:
    """从 State.messages 转成 {role, content} 列表。"""
    result = []
    for m in state.get("messages", []):
        role = "user" if isinstance(m, HumanMessage) else "assistant"
        result.append({"role": role, "content": m.content})
    return result


# 模块级模型实例（进程内复用）
_PRIMARY_LLM = None
_FALLBACK_LLM = None


def build_graph():
    """构建并编译 LangGraph 状态图。"""
    global _PRIMARY_LLM, _FALLBACK_LLM
    if _PRIMARY_LLM is None:
        _PRIMARY_LLM = config.get_llm()
        try:
            _FALLBACK_LLM = config.get_llm(use_fallback=True)
        except Exception:
            _FALLBACK_LLM = _PRIMARY_LLM

    g = StateGraph(AgentState)

    g.add_node("safety_check", safety_check)
    g.add_node("intent_router", intent_router)
    g.add_node("rag_retrieve", rag_retrieve)
    g.add_node("tool_execute", tool_execute)
    g.add_node("generate", generate)
    g.add_node("reject", reject)
    g.add_node("fallback", fallback)
    g.add_node("transfer_human", transfer_human)

    g.add_edge(START, "safety_check")
    g.add_conditional_edges("safety_check", _route_after_safety,
                            {"reject": "reject", "intent_router": "intent_router"})
    g.add_conditional_edges("intent_router", _route_after_intent, {
        "reject": "reject",
        "tool_execute": "tool_execute",
        "rag_retrieve": "rag_retrieve",
        "generate": "generate",
        "transfer_human": "transfer_human",
    })
    g.add_conditional_edges("rag_retrieve", _route_after_rag,
                            {"generate": "generate", "fallback": "fallback"})
    g.add_conditional_edges("tool_execute", _route_after_tool,
                            {"generate": "generate", "fallback": "fallback"})
    g.add_conditional_edges("fallback", _route_after_fallback,
                            {"transfer_human": "transfer_human", END: END})

    g.add_edge("generate", END)
    g.add_edge("reject", END)
    g.add_edge("transfer_human", END)

    return g.compile()


def _prepare_input(session_id: str, user_input: str,
                   messages: list | None = None) -> tuple[AgentState, str]:
    start = time.time()
    state_input: AgentState = {
        "session_id": session_id,
        "user_input": user_input,
        "filtered_input": user_input,
        "messages": messages or [],
        "clarify_round": 0,
        "fallback_type": "",
    }
    trace_id = f"conv_{session_id}_{int(start * 1000)}"
    return state_input, trace_id


def _finalize(session_id: str, result: dict, elapsed_ms: float) -> dict:
    """收尾：画像更新 + 指标记录 + 结果组装。"""
    intent = result.get("intent", "")
    try:
        if intent in ("order", "logistics"):
            memory.update_profile(session_id, topic="订单/物流")
        elif intent == "knowledge":
            memory.update_profile(session_id, topic="知识咨询")
    except Exception:
        pass

    observability.metrics.record(
        intent=intent,
        used_rag=bool(result.get("retrieved_docs")),
        fallback_type=result.get("fallback_type", ""),
        transfer=bool(result.get("transfer_package")),
        reject=result.get("fallback_type") == "reject" or
               result.get("response") == "",
        elapsed_ms=elapsed_ms,
    )

    return {
        "response": result.get("response", ""),
        "intent": intent,
        "intent_reason": result.get("intent_reason", ""),
        "retrieved_docs": result.get("retrieved_docs", []),
        "citations": result.get("citations", []),
        "fallback_type": result.get("fallback_type", ""),
        "transfer_package": result.get("transfer_package"),
        "used_fallback_model": result.get("used_fallback_model", False),
        "trace_id": result.get("_trace_id", ""),
        "elapsed_ms": round(elapsed_ms, 1),
    }


def run_agent(session_id: str, user_input: str,
              messages: list | None = None) -> dict:
    """执行一次完整对话，返回结果字典（供评估脚本等非流式场景复用）。"""
    graph = build_graph()
    start = time.time()
    state_input, trace_id = _prepare_input(session_id, user_input, messages)
    result = graph.invoke(state_input, config={"recursion_limit": 20})
    result["_trace_id"] = trace_id
    elapsed_ms = (time.time() - start) * 1000
    return _finalize(session_id, result, elapsed_ms)


def run_agent_stream(session_id: str, user_input: str,
                     messages: list | None = None):
    """流式执行：逐节点产出 (event_type, data)，供 SSE 输出。

    event_type:
      - node_done: 某节点完成，data 为 {node, ...摘要字段}
      - message: 最终回复就绪，data 为完整结果字典
    """
    graph = build_graph()
    start = time.time()
    state_input, trace_id = _prepare_input(session_id, user_input, messages)

    final_result = None
    extra_state: dict = {}
    for chunk in graph.stream(state_input, config={"recursion_limit": 20},
                              stream_mode="updates"):
        for node, update in chunk.items():
            # 进度事件（不含 response 的中间节点）
            if node in ("safety_check", "intent_router", "rag_retrieve",
                        "tool_execute", "fallback"):
                summary = {"node": node}
                if "intent" in update:
                    summary["intent"] = update["intent"]
                if "fallback_type" in update:
                    summary["fallback_type"] = update["fallback_type"]
                if "retrieved_docs" in update:
                    summary["doc_count"] = len(update["retrieved_docs"])
                yield "node_done", summary
            # 保留意图等关键状态（updates 模式只返回节点增量，需手动汇总）
            if node == "intent_router":
                extra_state["intent"] = update.get("intent", "")
                extra_state["intent_reason"] = update.get("intent_reason", "")
                extra_state["fallback_type"] = update.get("fallback_type", "")
            # 产出最终回复的节点：合并到 final_result（用 update 而非赋值，避免覆盖）
            if node in ("generate", "reject", "transfer_human", "fallback"):
                if update.get("response") or node == "transfer_human":
                    if final_result is None:
                        final_result = {}
                    final_result.update(update)

    if final_result is None:
        final_result = {"response": "抱歉，服务暂时不可用，请稍后重试。"}
    final_result.update(extra_state)
    final_result["_trace_id"] = trace_id
    elapsed_ms = (time.time() - start) * 1000
    yield "message", _finalize(session_id, final_result, elapsed_ms)
