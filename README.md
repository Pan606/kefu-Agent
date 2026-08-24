# 智能客服 Agent

基于 LangGraph + FastAPI + RAG 的智能客服 Agent（V1.0 初版），覆盖设计文档《智能客服Agent设计文档_v2.docx》的 M0–M4 全部里程碑。

## 功能

- **多轮对话**：基于会话上下文连续应答，支持澄清追问
- **知识问答（RAG）**：基于演示知识库（42 个问答对 / 5 类主题）检索回答，附来源引用
- **业务工具**：订单状态 / 物流轨迹查询（V1 模拟数据，接口按真实系统标准设计）
- **违禁识别**：敏感词过滤 + 提示注入检测 + 委婉拒绝（L3）
- **兜底策略**：L1 澄清（最多 2 轮）→ L2 转人工（携带交接信息包）
- **会话持久化**：SQLite 存储，服务重启不丢上下文，Web 端可恢复历史
- **用户画像**：偏好 / 主题统计 / 语气，按会话读写（V1 用 session_id 代替 user_id）
- **满意度评价**：每次回复可点「有用 / 没用」
- **可观测性**：Langfuse 追踪（未配置 key 自动降级）+ 核心指标接口

## 技术栈

| 组件 | 选型 |
|---|---|
| 对话模型 | 智谱 glm-4-flash（主）/ DeepSeek deepseek-v4-flash（备，自动降级） |
| Embedding | 智谱 embedding-3（2048 维） |
| 编排框架 | LangGraph（LangChain 1.x） |
| 后端 | FastAPI + uvicorn + SSE 流式 |
| 向量库 | Chroma（本地持久化） |
| 存储 | SQLite（会话 / 画像 / 评价 / 交接包） |
| 前端 | 原生 HTML+JS（无构建链） |

## 快速开始

```bash
# 1. 安装依赖（conda 环境 Python 3.12）
conda create -n kefu_agent python=3.12 -y
conda activate kefu_agent
pip install -r requirements.txt

# 2. 配置 API key
cp .env.example .env   # 填入 ZHIPU_API_KEY / DEEPSEEK_API_KEY

# 3. 生成演示数据 + 构建知识库索引 + 启动
python scripts/seed_data.py   # 生成 FAQ 与模拟订单
python scripts/init_kb.py     # 构建 Chroma 索引
python main.py                # 启动 → http://127.0.0.1:8761
```

## 目录结构

```
app/
  config.py          # 配置 + 模型抽象层（主/备切换）
  safety.py          # 敏感词 / 注入检测 / 脱敏
  intent_router.py   # 意图分类（模型 + 规则兜底）
  rag_engine.py      # Chroma 检索 + 阈值过滤 + 引用
  tool_registry.py   # 工具注册表（query_order / query_logistics / transfer_human）
  memory.py          # SQLite 会话 / 画像 / 评价 / 交接包
  agent_graph.py     # LangGraph 状态图（节点 + 条件边）
  observability.py   # Langfuse（可降级）+ 指标
  web_server.py      # FastAPI + SSE
scripts/
  seed_data.py       # 演示数据生成
  init_kb.py         # 知识库索引
web/index.html       # 聊天页
tests/eval_set.py    # 22 题评估集
```

## LangGraph 状态图

```
START → safety_check ──拦截──→ reject(L3) → END
            │ 通过
            ▼
       intent_router ──违禁──→ reject
            │ 知识问答          ──订单/物流──→ tool_execute ──成功──→ generate → END
            │                    │                          │ 失败
            ▼                    │                          ▼
       rag_retrieve ──命中──→ generate ─────────────→ END  fallback
            │ 无命中                                          │
            ▼                                                ▼
       fallback(L1 澄清≤2轮) ──仍无解──→ transfer_human(L2) → END
```

## 接口

| 接口 | 方法 | 说明 |
|---|---|---|
| `/api/chat` | POST | 对话请求，SSE 流式返回 |
| `/api/session` | POST | 新建会话 |
| `/api/session/{id}` | GET | 会话恢复 |
| `/api/feedback` | POST | 满意度评价 |
| `/api/transfer` | POST | 转人工 |
| `/api/metrics` | GET | 核心指标快照 |
| `/health` | GET | 健康检查 |

## 评估

```bash
python tests/eval_set.py --run
```

22 题四类用例（normal 9 / boundary 5 / forbidden 4 / no_answer 4）。违禁类必须全部通过（安全底线）。检索参数（Top-K / 阈值）在 `config.py` 或 `.env` 调整，配合评估集调参。

## 后续迭代（未纳入 V1.0）

- Token 级流式输出（当前为节点级进度 + 完整回复）
- 微信 / 企微渠道接入
- pgvector / PostgreSQL 迁移（远程服务器 119.45.120.52）
- Docker 化部署
- 真实业务系统对接（替换 tool_registry 模拟实现）
