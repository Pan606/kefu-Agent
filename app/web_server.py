"""Web 服务：FastAPI + SSE 流式对话 + 静态聊天页（对应文档 4.2 web_server / 11.2）。

接口：
  GET  /                 → 聊天页
  POST /api/chat         → 对话请求（SSE 流式）
  POST /api/session      → 新建会话
  GET  /api/session/{id} → 会话恢复
  POST /api/feedback     → 满意度评价
  POST /api/transfer     → 转人工（V1 记录交接包）
  GET  /api/metrics      → 核心指标快照
  GET  /health           → 健康检查
"""
from __future__ import annotations

import asyncio
import time
import uuid
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import agent_graph, config, memory, observability, safety
from .observability import logger

app = FastAPI(title="智能客服 Agent", version="0.1.0")

# ---------- 请求模型 ----------

class ChatRequest(BaseModel):
    session_id: str | None = None
    message: str = Field(..., min_length=1, max_length=2000)


class FeedbackRequest(BaseModel):
    session_id: str
    message_id: str
    rating: str  # useful / useless


class TransferRequest(BaseModel):
    session_id: str


class SessionResponse(BaseModel):
    session_id: str


# ---------- 会话管理 ----------

def _new_session_id() -> str:
    return f"S{int(time.time())}{uuid.uuid4().hex[:8]}"


def _ensure_session(session_id: str | None) -> str:
    if session_id:
        memory.create_session(session_id)  # INSERT OR IGNORE
        return session_id
    sid = _new_session_id()
    memory.create_session(sid)
    return sid


# ---------- 路由 ----------

@app.get("/health")
async def health():
    return {"status": "ok", "time": time.strftime("%Y-%m-%d %H:%M:%S")}


@app.post("/api/session", response_model=SessionResponse)
async def create_session():
    sid = _new_session_id()
    memory.create_session(sid)
    return {"session_id": sid}


@app.get("/api/session/{session_id}")
async def get_session(session_id: str):
    messages = memory.load_session(session_id)
    if not messages:
        raise HTTPException(status_code=404, detail="会话不存在")
    return {"session_id": session_id, "messages": messages}


@app.post("/api/chat")
async def chat(req: ChatRequest):
    """SSE 流式对话：节点进度事件 + 最终回复事件。"""
    session_id = _ensure_session(req.session_id)

    # 请求级限流（简单内存令牌桶：每会话最多 5 并发）
    if _rate_limited(session_id):
        return JSONResponse(
            status_code=429,
            content={"detail": "请求过于频繁，请稍后再试"})

    history = memory.get_history_messages(session_id)

    async def event_generator():
        # 1) 记录用户消息
        memory.append_message(session_id, "user", req.message)
        try:
            # 2) 流式跑图：先发节点进度，再发最终回复
            async for event_type, data in _stream_run(session_id, req.message, history):
                if event_type == "node_done":
                    yield f"event: node\ndata: {_json(data)}\n\n"
                else:
                    # 持久化回复 + 返回给前端
                    memory.append_message(session_id, "assistant", data["response"])
                    yield f"event: message\ndata: {_json(data)}\n\n"
            yield "event: done\ndata: {}\n\n"
        except Exception as e:
            logger.exception("chat 处理异常")
            err = {"error": str(e)[:200]}
            yield f"event: error\ndata: {_json(err)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


def _next_or_none(gen):
    """线程内执行 next()，StopIteration 转 None（规避 Future 内 raise 限制）。"""
    try:
        return next(gen)
    except StopIteration:
        return None


async def _stream_run(session_id: str, message: str, history: list[dict]):
    """把 agent_graph 的同步生成器转成异步生成器。"""
    gen = agent_graph.run_agent_stream(session_id, message, history)
    loop = asyncio.get_running_loop()
    while True:
        item = await loop.run_in_executor(None, _next_or_none, gen)
        if item is None:
            break
        yield item


@app.post("/api/feedback")
async def feedback(req: FeedbackRequest):
    if req.rating not in ("useful", "useless"):
        raise HTTPException(status_code=400, detail="rating 必须为 useful/useless")
    memory.save_feedback(req.session_id, req.message_id, req.rating)
    logger.info("feedback: %s %s", req.session_id, req.rating)
    return {"ok": True}


@app.post("/api/transfer")
async def transfer(req: TransferRequest):
    """V1 转人工：立即记录交接包（信息来自最近会话状态）。"""
    memory.create_session(req.session_id)
    profile = memory.get_profile(req.session_id)
    history = memory.get_history_messages(req.session_id)
    handoff = {
        "session_id": req.session_id,
        "summary": f"用户主动转人工，最近 {len(history)} 条消息",
        "profile": {"preferences": profile["preferences"],
                    "topics": profile["topics"]},
        "intent": "user_request",
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    memory.save_transfer(req.session_id, handoff)
    return {"ok": True, "handoff": handoff}


@app.get("/api/metrics")
async def metrics():
    return observability.metrics.snapshot()


# ---------- 静态页面 ----------

@app.get("/", include_in_schema=False)
async def index():
    return FileResponse(config.WEB_DIR / "index.html")


app.mount("/static", StaticFiles(directory=str(config.WEB_DIR)), name="static")


# ---------- 限流（内存令牌桶，演示级） ----------

_RATE: dict[str, list[float]] = {}


def _rate_limited(session_id: str, max_per_minute: int = 20) -> bool:
    now = time.time()
    _RATE.setdefault(session_id, []).append(now)
    _RATE[session_id] = [t for t in _RATE[session_id] if now - t < 60]
    return len(_RATE[session_id]) > max_per_minute


def _json(data: dict) -> str:
    import json
    return json.dumps(data, ensure_ascii=False)


def main() -> None:
    """入口：初始化数据库与可观测性后启动服务。"""
    memory.init_db()
    observability.init_observability()
    logger.info("智能客服 Agent 启动于 http://127.0.0.1:8761")
    uvicorn.run(app, host="127.0.0.1", port=8761)


if __name__ == "__main__":
    main()
