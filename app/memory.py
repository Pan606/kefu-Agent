"""记忆模块：SQLite 会话持久化 + 用户画像读写（对应文档第 7 章）。

设计要点（对应文档 7.1 关键设计）：
- 三类数据分开存储：知识库(向量) / 用户画像(SQLite) / 会话记录(SQLite)；
- 短期记忆：会话消息窗口，恢复时按最近 N 轮裁剪（config.MAX_HISTORY_ROUNDS）；
- 长期画像：preferences / topics / tone / last_seen，按 user_id 读写；
- V1 用 session_id 代替 user_id（匿名用户，对应文档 7.3）。
"""
from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path

from . import config

_DB_INIT_SQL = """
CREATE TABLE IF NOT EXISTS sessions (
    session_id  TEXT PRIMARY KEY,
    messages    TEXT NOT NULL,          -- JSON 数组 [{role, content, ts}]
    created_at  REAL NOT NULL,
    updated_at  REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS user_profiles (
    user_id     TEXT PRIMARY KEY,       -- V1 用 session_id 代替
    preferences TEXT NOT NULL DEFAULT '[]',  -- JSON 数组
    topics      TEXT NOT NULL DEFAULT '{}',  -- JSON 对象 {topic: count}
    tone        TEXT NOT NULL DEFAULT '',
    last_seen   REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS feedback (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  TEXT NOT NULL,
    message_id  TEXT NOT NULL,
    rating      TEXT NOT NULL,          -- useful / useless
    created_at  REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS transfers (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  TEXT NOT NULL,
    handoff     TEXT NOT NULL,          -- 交接信息包 JSON
    created_at  REAL NOT NULL
);
"""


def _get_conn() -> sqlite3.Connection:
    config.STORE_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(config.DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """初始化数据库表结构（幂等）。"""
    conn = _get_conn()
    try:
        conn.executescript(_DB_INIT_SQL)
        conn.commit()
    finally:
        conn.close()


# ---------- 会话 ----------

def create_session(session_id: str) -> None:
    now = time.time()
    conn = _get_conn()
    try:
        conn.execute(
            "INSERT OR IGNORE INTO sessions (session_id, messages, created_at, updated_at)"
            " VALUES (?, ?, ?, ?)",
            (session_id, json.dumps([], ensure_ascii=False), now, now))
        conn.execute(
            "INSERT OR IGNORE INTO user_profiles (user_id, last_seen) VALUES (?, ?)",
            (session_id, now))
        conn.commit()
    finally:
        conn.close()


def load_session(session_id: str) -> list[dict]:
    """加载会话消息；不存在返回空列表。"""
    conn = _get_conn()
    try:
        row = conn.execute(
            "SELECT messages FROM sessions WHERE session_id = ?",
            (session_id,)).fetchone()
        if not row:
            return []
        return json.loads(row["messages"])
    finally:
        conn.close()


def append_message(session_id: str, role: str, content: str) -> None:
    """追加一条消息并裁剪到最近 N 轮。"""
    messages = load_session(session_id)
    messages.append({"role": role, "content": content, "ts": time.time()})
    # 裁剪：保留最近 MAX_HISTORY_ROUNDS 轮（1 轮 = 1 问 + 1 答）
    max_msgs = config.MAX_HISTORY_ROUNDS * 2
    if len(messages) > max_msgs:
        messages = messages[-max_msgs:]
    conn = _get_conn()
    try:
        conn.execute(
            "UPDATE sessions SET messages = ?, updated_at = ? WHERE session_id = ?",
            (json.dumps(messages, ensure_ascii=False), time.time(), session_id))
        conn.commit()
    finally:
        conn.close()


def get_history_messages(session_id: str) -> list[dict]:
    """返回用于提示词的消息列表（不含 ts）。"""
    return [
        {"role": m["role"], "content": m["content"]}
        for m in load_session(session_id)
    ]


# ---------- 用户画像 ----------

def get_profile(user_id: str) -> dict:
    conn = _get_conn()
    try:
        row = conn.execute(
            "SELECT preferences, topics, tone, last_seen FROM user_profiles WHERE user_id = ?",
            (user_id,)).fetchone()
        if not row:
            return {"preferences": [], "topics": {}, "tone": "", "last_seen": 0}
        return {
            "preferences": json.loads(row["preferences"]),
            "topics": json.loads(row["topics"]),
            "tone": row["tone"],
            "last_seen": row["last_seen"],
        }
    finally:
        conn.close()


def update_profile(user_id: str, preference: str | None = None,
                   topic: str | None = None, tone: str | None = None) -> None:
    """画像更新：偏好追加、主题计数、语气更新、活跃时间刷新。"""
    profile = get_profile(user_id)
    if preference and preference not in profile["preferences"]:
        profile["preferences"].append(preference)
    if topic:
        profile["topics"][topic] = profile["topics"].get(topic, 0) + 1
    if tone:
        profile["tone"] = tone
    conn = _get_conn()
    try:
        conn.execute(
            "INSERT INTO user_profiles (user_id, preferences, topics, tone, last_seen)"
            " VALUES (?, ?, ?, ?, ?)"
            " ON CONFLICT(user_id) DO UPDATE SET"
            " preferences=excluded.preferences, topics=excluded.topics,"
            " tone=excluded.tone, last_seen=excluded.last_seen",
            (user_id,
             json.dumps(profile["preferences"], ensure_ascii=False),
             json.dumps(profile["topics"], ensure_ascii=False),
             profile["tone"], time.time()))
        conn.commit()
    finally:
        conn.close()


# ---------- 满意度评价 ----------

def save_feedback(session_id: str, message_id: str, rating: str) -> None:
    conn = _get_conn()
    try:
        conn.execute(
            "INSERT INTO feedback (session_id, message_id, rating, created_at)"
            " VALUES (?, ?, ?, ?)",
            (session_id, message_id, rating, time.time()))
        conn.commit()
    finally:
        conn.close()


# ---------- 转人工交接 ----------

def save_transfer(session_id: str, handoff: dict) -> None:
    conn = _get_conn()
    try:
        conn.execute(
            "INSERT INTO transfers (session_id, handoff, created_at) VALUES (?, ?, ?)",
            (session_id, json.dumps(handoff, ensure_ascii=False), time.time()))
        conn.commit()
    finally:
        conn.close()


def get_transfers(session_id: str) -> list[dict]:
    conn = _get_conn()
    try:
        rows = conn.execute(
            "SELECT handoff FROM transfers WHERE session_id = ? ORDER BY id DESC",
            (session_id,)).fetchall()
        return [json.loads(r["handoff"]) for r in rows]
    finally:
        conn.close()
