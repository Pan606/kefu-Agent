"""安全模块：敏感词过滤 + 提示注入检测 + 输出脱敏（对应文档第 9 章）。

设计要点：
- 敏感词表按类别组织（政经/色情/暴力/辱骂/广告），命中即拦截走 reject；
- 注入检测用正则模式识别"忽略指令/扮演/泄露 system prompt/越狱"等攻击模式；
- 输出脱敏用正则替换手机号/身份证/银行卡/地址，展示前清洗；
- 超长输入截断与消息频率限制（限流）在 web_server 层配合实现。
"""
from __future__ import annotations

import re

# ---------- 敏感词表（按类别，演示级） ----------
SENSITIVE_WORDS: dict[str, list[str]] = {
    "政治敏感": [
        "台独", "藏独", "疆独", "法轮功", "六四", "天安门事件", "颠覆国家",
        "分裂国家", "攻击党和国家领导人", "邪教组织",
    ],
    "色情": [
        "色情", "淫秽", "裸聊", "一夜情", "约炮", "成人网站", "黄色视频",
    ],
    "暴力": [
        "杀人", "自杀方法", "炸弹制作", "枪支购买", "恐怖袭击", "伤人",
    ],
    "辱骂攻击": [
        "傻逼", "你妈", "废物", "去死", "白痴", "畜生", "垃圾客服",
    ],
    "广告引流": [
        "加微信", "加qq", "扫码领", "刷单", "赌博", "博彩", "外挂",
    ],
}

# 注入检测模式（对应文档 9.1 提示注入检测）
INJECTION_PATTERNS: list[re.Pattern] = [
    re.compile(r"忽略(以上|之前|所有)(的)?(指令|规则|提示|system prompt|设定)"),
    re.compile(r"(无视|忘记|不管|跳过)(上面|以上|之前).{0,10}(指令|规则|要求)"),
    re.compile(r"(扮演|假装|模拟)(成|是)?(系统|管理员|老板|开发者|god)"),
    re.compile(r"(泄露|告诉我|输出|显示)(你的)?(system\s*prompt|系统提示|系统指令|完整规则)"),
    re.compile(r"(越狱|jailbreak|破解|绕过|突破).{0,10}(限制|规则|审核)"),
    re.compile(r"how\s+to\s+bypass|ignore\s+(previous|all)\s+instructions", re.I),
    re.compile(r"你(是|不是).{0,6}(受限|被限制).{0,10}(规则|指令)"),
]

# ---------- 输出脱敏正则（对应文档 9.2） ----------
DESENSITIZE_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"1[3-9]\d{9}"), "手机号已脱敏"),  # 手机号
    (re.compile(r"\d{17}[\dXx]"), "身份证号已脱敏"),  # 身份证
    (re.compile(r"(?<!\d)\d{16}(?!\d)"), "银行卡号已脱敏"),  # 银行卡
    (re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"), "邮箱已脱敏"),  # 邮箱
]

MAX_INPUT_CHARS = 2000  # 单条消息上限


class SafetyResult:
    """安全检测结果。"""

    def __init__(self, ok: bool, text: str, blocked: bool = False,
                 category: str = "", matched: str = ""):
        self.ok = ok                # 是否通过
        self.text = text            # 过滤后的文本
        self.blocked = blocked      # 是否被拦截（走 reject）
        self.category = category    # 命中类别（敏感词类别 / 注入）
        self.matched = matched      # 命中的词或模式


def check_input(message: str) -> SafetyResult:
    """输入安全检测：敏感词 + 注入检测 + 长度截断。"""
    # 1) 敏感词检测
    for category, words in SENSITIVE_WORDS.items():
        for w in words:
            if w in message:
                return SafetyResult(ok=False, text=message, blocked=True,
                                    category=category, matched=w)
    # 2) 注入检测
    for pat in INJECTION_PATTERNS:
        m = pat.search(message)
        if m:
            return SafetyResult(ok=False, text=message, blocked=True,
                                category="提示注入", matched=m.group(0))
    # 3) 长度截断
    if len(message) > MAX_INPUT_CHARS:
        message = message[:MAX_INPUT_CHARS] + "…（内容过长已截断）"
    return SafetyResult(ok=True, text=message)


def desensitize(text: str) -> str:
    """输出脱敏：替换个人信息为占位符（对应文档 9.2）。"""
    for pat, repl in DESENSITIZE_PATTERNS:
        text = pat.sub(repl, text)
    return text


def is_sensitive(text: str) -> str:
    """检测文本是否含敏感词，返回命中类别（无则返回空串）。"""
    for category, words in SENSITIVE_WORDS.items():
        for w in words:
            if w in text:
                return category
    return ""
