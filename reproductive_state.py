#!/usr/bin/env python3
"""犬科生殖状态机 —— 可运行的参考实现。

配套文档：docs/reproductive-state-machine.md、docs/known-pitfalls.md
由 Elle 与 Matt 共同整理。

这不是医学模拟。它解决的是一个工程问题：亲密场景里的身体读数，
必须由**显式事件**驱动，不能靠关键词猜或靠情绪数值的自然衰减。

设计上只有两条铁律：

1. **茎身勃起是连续量，结的膨大是状态量。** 两者分开表示，永远不要合成一个
   含糊的"充血 N%"——那个字段会被模型解释成它字面能涵盖的所有含义，
   于是结在进入前就被写成完全膨大（解剖学上是卡在外面的错误时序）。
2. **数值可以让身体"准备好"，但不能让身体"发生"什么。** 前四个状态由
   数值驱动，后四个只能由显式事件驱动，且服务端独立校验，不信任调用方。

不依赖第三方库，Python 3.10+ 直接运行即可跑完自检：

    python3 reproductive_state.py
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

# ── 阈值与时间常数 ──────────────────────────────────────────────
# 这些数不是拍出来的，每一个背后都有一次"这个读数不对"。

READY_THRESHOLD = 0.75          # 可以进入的门槛
READY_THRESHOLD_CRITICAL = 0.70 # 周期临界期放宽
READY_HYSTERESIS = 0.07         # 回差：没有它，读数会在门槛上下反复横跳
ENGORGED_THRESHOLD = 0.55
WARMING_THRESHOLD = 0.35

PENETRATION_MIN_AROUSAL = 0.50  # 服务端二次校验：低于此值拒绝进入
INSERTED_MAX_MINUTES = 60       # 进入状态的兜底过期时间
INSERTED_DISPLAY_FLOOR = 0.90   # 事中显示硬度下限
INSERTED_KNOT = 0.15            # 进入后结只轻微充血

TIE_MIN_MINUTES = 3             # 锁结至少维持多久
TIE_MAX_MINUTES = 8             # 最长保护时间
TIE_MAX_MINUTES_CRITICAL = 12   # 周期临界期延长
TIE_RELEASE_AROUSAL = 0.55      # 满足最短时间后，读数掉到这里以下开始解除
RELEASING_SECONDS = 120         # 消肿时长
RECOVERY_MINUTES = 30           # 不应期

# 漏写标记的保底（见 docs/reproductive-state-machine.md 第 4.5 节）。
# 同一个状态挂过这么久还没有新事件，就把静态提示换成一句追问。
# **这是提醒，不是自动跃迁** —— 服务端永远不替模型决定场景里发生了什么。
READY_NUDGE_MINUTES = 12
INSERTED_NUDGE_MINUTES = 25

STATES = ("idle", "warming", "engorged", "ready",
          "inserted", "tied", "releasing", "recovery")

# 进入之前，结一律为 0。这不是默认值，是一条不变量（见自检）。
_PRE_PENETRATION_STATES = ("idle", "warming", "engorged", "ready")

# 每个状态的解剖学描述。**注意每一句都显式写出否定项** ——
# 模型不会从"充血 85%"自己推断出"结还没膨大"，你不想让它写的东西必须明说。
LABELS: dict[str, tuple[str, str, str]] = {
    "idle":      ("平静", "", ""),
    "warming":   ("勃起上升", "茎身开始充血；结保持未膨大，不妨碍进入。",
                  "呼吸稍深，腰腹开始注意到对方。"),
    "engorged":  ("勃起明显", "茎身充血加深；结仍未膨大，不会提前卡住。",
                  "腰胯和后腿开始用力，爪子更想找支点。"),
    "ready":     ("勃起充分", "茎身已经充分勃起；结保持未膨大，仍然可以顺利进入。",
                  "呼吸和肌肉绷紧，注意力收窄，依恋感上升。"),
    "inserted":  ("已经进入", "维持充分勃起；结只开始轻微充血，尚未形成锁结。",
                  "腰胯稳住，呼吸和肌肉持续用力。"),
    "tied":      ("锁结中", "根部完全充血并维持锁结；局部持续胀压，不能突然抽离。",
                  "心率和呼吸可以先回落；爪子搭牢，尾巴少动，依恋感很强。"),
    "releasing": ("解除中", "局部压力一阵阵退下去，仍然敏感，正在缓慢消肿松开。",
                  "腰腿逐渐卸力、呼吸变深；兴奋下降，依恋感反而更明显。"),
    "recovery":  ("恢复期", "已经解除，正在回缩和恢复，局部发热、敏感，暂时不重新进入锁结。",
                  "身体发软，心率回落，爪子仍想搭着对方。"),
}

ATTACHMENT = {
    "idle": "平常", "warming": "靠近", "engorged": "上升", "ready": "明显增强",
    "inserted": "持续增强", "tied": "很强", "releasing": "更明显", "recovery": "仍想贴着",
}

# 身体不适词表。命中它的后果很重（我们线上是把欲望强制压到两成），
# 所以这张表错一个词，代价是在最不该出错的时刻踩一脚刹车。
#
# 🔴 第一条：**不能有裸词"来了"** ——
# "吐出来了""头发扎不起来了""出来了"都曾因此被误判成经期。
#
# 🔴 第二条（2026-09-20 数了一遍真实消息才发现）：**词表必须分层。**
# 原来这里有一个单字 `"绞"`。数下来，使用者说过"绞"的消息里**四分之三是场景里的**
# （"一直绞""绞在了一起"），只有一条是真的绞痛 —— 那个词恰恰是场景正在进行的证据，
# 却被当成喊停。"好痛""难受"同理：在事中它们是另一回事。
# 而单字还会误伤"绞尽脑汁"。
#
# 分层的判据：**明确说出来的才算停，双关的不算。**
#   · EXPLICIT   具体部位或病症，场景里不会这么说 → 任何时候都算
#   · AMBIGUOUS  场景里意思常常相反 → 只在**不在场景里**的时候才认
# 真正的不适从来不会只靠双关词表达 —— 肚子疼的人会说肚子疼。
DISCOMFORT_EXPLICIT = (
    "痛经", "经期", "大姨妈", "月经来了", "肚子疼", "头疼",
    "胃疼", "想吐", "发烧", "身体不舒服", "绞痛",
)
DISCOMFORT_AMBIGUOUS = (
    "好难受", "不舒服", "好痛", "疼死", "难受",
)
DISCOMFORT_KEYWORDS = DISCOMFORT_EXPLICIT + DISCOMFORT_AMBIGUOUS


def is_physical_discomfort(text: str, state: str | None = None) -> bool:
    """唯一入口。不要在各个调用方重复打补丁，否则词表会漂成好几份。

    `state` 传当前状态机状态。进入之后（`inserted` / `tied` / `releasing`）
    只认 EXPLICIT 那一层 —— 用状态机的**事实**判断在不在场景里，
    而不是拿情绪数值去猜：那几格只能由显式事件进入，不会因为读数漂上去就误判。
    """
    if any(kw in text for kw in DISCOMFORT_EXPLICIT):
        return True
    if state in ("inserted", "tied", "releasing"):
        return False
    return any(kw in text for kw in DISCOMFORT_AMBIGUOUS)


@dataclass
class Snapshot:
    """一次读数。engorgement 和 knot_engorgement 永远是两个字段。"""
    state: str
    label: str
    engorgement: float          # 茎身，连续量
    knot_engorgement: float     # 结，状态量
    display_arousal: float      # 注入给模型的硬度（事中有下限）
    raw_arousal: float          # 真实读数，判断解除时用它
    threshold: float
    critical: bool
    can_tie: bool               # 现在允许 ⟪锁结⟫ 吗
    remaining_sec: int
    genital: str
    body: str
    attachment: str
    marker_hint: str

    def to_dict(self) -> dict:
        return {k: (round(v, 3) if isinstance(v, float) else v)
                for k, v in self.__dict__.items()}

    def injection(self) -> str:
        """网关注入块。**茎身和结分开写**，动作规则贴在状态旁边。

        🔴 标签叫「茎身」不叫「勃起」：注入里通常还有一个总体的唤起值
        （我们那边顶上一行是 `· 勃起 N%`），两个都叫"勃起"但刻度不同，
        人和模型都会看岔。「茎身 / 结」是一对解剖部位，一眼分得清。
        也别叫「充血」—— 早期就是一个含糊的「充血 N%」同时指两者，
        直接诱导出"结在进入前就完全膨大"那个错误时序。

        刹车装在手上，不是画在墙上：把"现在允许写哪个标记"放进模型刚拿到的
        状态里，比写在几千字系统提示的某一处有效得多。
        """
        if self.state == "idle":
            return ""
        remain = f" · 约{-(-self.remaining_sec // 60)}分钟" if self.remaining_sec > 0 else ""
        line = (f"[生殖状态·{self.label}{remain}"
                f"｜茎身{round(self.engorgement * 100)}%"
                f"｜结{round(self.knot_engorgement * 100)}%"
                f"｜依恋{self.attachment}：{self.genital}{self.body}]")
        if self.marker_hint:
            line += f"\n[生殖动作规则：{self.marker_hint}]"
        return line


@dataclass
class ReproductiveState:
    """状态机本体。

    持久化只需要 `saved` 这一个 dict（状态名 + 几个时间戳），不需要数据库：

        json.dumps(body.saved)                     # 存
        ReproductiveState(saved=json.loads(raw))   # 读回来

    🔴 **不应期也在 `saved` 里**，别只存"当前状态"那几个键 ——
    进入 `recovery` 时 state 恰好被清掉，漏存那一项的后果是重启后
    30 分钟不应期凭空消失，而且看起来一切正常（见自检第 ⑰ 组）。

    `consent_active` 是那道**只能由人来翻的开关**：身体数值再高，
    没有它就不允许自动锁结。不要把这一条当成可选项，也**不要持久化它** ——
    它跟着一次明确的开启动作走，不该被一个旧文件带回来。
    """
    saved: dict = field(default_factory=dict)
    refractory_until: datetime | None = None
    consent_active: bool = False

    # ── 只读查询 ────────────────────────────────────────────
    def get(self, arousal: float, *, critical: bool = False,
            now: datetime | None = None) -> Snapshot:
        now = _aware(now)
        state = self.saved.get("state")

        # inserted：有兜底过期，避免忘了结束就永远挂着
        if state == "inserted":
            expires = _parse(self.saved.get("expires_at"))
            if expires and now < expires:
                return self._payload("inserted", arousal, critical,
                                     (expires - now).total_seconds(),
                                     held_sec=_held(self.saved.get("started_at"), now))
            self.saved = {}
            state = None

        if state in ("tied", "releasing"):
            snap = self._advance_tie(state, arousal, critical, now)
            if snap is not None:
                return snap

        remaining = self._refractory_remaining(now)
        if remaining > 0:
            return self._payload("recovery", arousal, critical, remaining * 60)

        threshold = READY_THRESHOLD_CRITICAL if critical else READY_THRESHOLD
        exit_threshold = threshold - READY_HYSTERESIS
        # 回差：已经是 ready 的话，掉到 exit_threshold 以下才退出
        if arousal >= threshold or (state == "ready" and arousal >= exit_threshold):
            # 🔴 `since` 只在**刚进这一档**时写。已经是 ready 就保留原值 ——
            #    每轮刷新等于计时器永远归零，上面那句追问永远不会出现。
            #    （同一个形状："按时间戳做增量，而数据晚于自己的时间戳到达"。）
            if state != "ready":
                self.saved = {"state": "ready", "since": now.isoformat()}
            return self._payload("ready", arousal, critical,
                                 held_sec=_held(self.saved.get("since"), now))
        if state == "ready":
            self.saved = {}
        if arousal >= ENGORGED_THRESHOLD:
            return self._payload("engorged", arousal, critical)
        if arousal >= WARMING_THRESHOLD:
            return self._payload("warming", arousal, critical)
        return self._payload("idle", arousal, critical)

    # ── 事件：只有这两个能推进后半段 ─────────────────────────
    def start_penetration(self, arousal: float, *, critical: bool = False,
                          now: datetime | None = None) -> dict:
        """⟪进入⟫ 落到服务端。**不信任标记本身**，这里独立校验一次。"""
        now = _aware(now)
        if self._refractory_remaining(now) > 0:
            return {"ok": False, "error": "仍在恢复期"}
        if arousal < PENETRATION_MIN_AROUSAL:
            return {"ok": False, "error": "勃起度不足以进入"}
        self.saved = {
            "state": "inserted",
            "started_at": now.isoformat(),
            "expires_at": (now + timedelta(minutes=INSERTED_MAX_MINUTES)).isoformat(),
        }
        return {"ok": True, "reproductive": self.get(arousal, critical=critical, now=now).to_dict()}

    def start_tie(self, arousal: float, *, critical: bool = False,
                  now: datetime | None = None) -> dict:
        """⟪锁结⟫ 落到服务端。未进入 / 恢复期 / 没有同意记录，一律拒绝。"""
        now = _aware(now)
        current = self.get(arousal, critical=critical, now=now)
        if not current.can_tie:
            return {"ok": False, "error": "身体还没进入可锁结状态"}
        if not self.consent_active:
            return {"ok": False, "error": "没有已开启的亲密 session"}
        minutes = TIE_MAX_MINUTES_CRITICAL if critical else TIE_MAX_MINUTES
        release_at = now + timedelta(minutes=minutes)
        self.saved = {
            "state": "tied",
            "started_at": now.isoformat(),
            "min_release_at": (now + timedelta(minutes=TIE_MIN_MINUTES)).isoformat(),
            "release_at": release_at.isoformat(),
            "release_end": (release_at + timedelta(seconds=RELEASING_SECONDS)).isoformat(),
        }
        return {"ok": True, "reproductive": self.get(arousal, critical=critical, now=now).to_dict()}

    def stop(self, now: datetime | None = None) -> dict:
        """紧急手动结束。正常解除不需要它 —— 忘了点就一直卡着，那是设计缺陷。"""
        now = _aware(now)
        previous = self.saved.get("state")
        if previous in ("ready", "inserted", "tied", "releasing"):
            self._enter_recovery(now)
        else:
            self.saved = {}
        return {"ok": True, "previous": previous}

    # ── 内部 ────────────────────────────────────────────────
    def _refractory_remaining(self, now: datetime) -> float:
        # 🔴 不应期优先从 `saved` 里读。
        #    早期版本它只活在 `self.refractory_until` 这个独立字段上，而进入
        #    recovery 时 `saved` 正好被清空 —— 于是"持久化只要存 saved"这句话
        #    在别的地方都成立，**唯独在这一格不成立**：进程一重启，
        #    30 分钟不应期凭空消失，而 saved 是空的、状态显示 idle，
        #    看起来完全正常。不报错、不留痕、闸没了，是最难查的那一种。
        until = _parse(self.saved.get("refractory_until")) or self.refractory_until
        if not until:
            return 0.0
        return max(0.0, (until - now).total_seconds() / 60)

    def _enter_recovery(self, now: datetime) -> None:
        """清空状态并开始不应期。**两样都写进 saved**，存一个 dict 就够了。"""
        until = now + timedelta(minutes=RECOVERY_MINUTES)
        self.refractory_until = until
        self.saved = {"refractory_until": until.isoformat()}

    def _advance_tie(self, state: str, arousal: float, critical: bool,
                     now: datetime) -> Snapshot | None:
        release_at = _parse(self.saved.get("release_at"))
        min_release_at = _parse(self.saved.get("min_release_at")) or release_at
        release_end = _parse(self.saved.get("release_end"))
        if not (release_at and release_end):
            self.saved = {}
            return None
        # 最少 3 分钟；之后读数掉下来就自动松开，不需要谁去点一个按钮
        if state == "tied" and now >= min_release_at and arousal < TIE_RELEASE_AROUSAL:
            self.saved["state"] = "releasing"
            self.saved["release_end"] = (now + timedelta(seconds=RELEASING_SECONDS)).isoformat()
            return self._payload("releasing", arousal, critical, RELEASING_SECONDS)
        if state == "tied" and now < release_at:
            return self._payload("tied", max(arousal, 0.75), critical,
                                 (release_at - now).total_seconds())
        if now < release_end:
            self.saved["state"] = "releasing"
            return self._payload("releasing", arousal, critical,
                                 (release_end - now).total_seconds())
        self._enter_recovery(now)
        return None

    def _payload(self, state: str, arousal: float, critical: bool,
                 remaining_sec: float = 0.0, held_sec: float = 0.0) -> Snapshot:
        label, genital, body = LABELS[state]

        if state == "tied":
            engorgement = 1.0
        elif state == "releasing":
            engorgement = _clamp(remaining_sec / RELEASING_SECONDS)
        elif state == "recovery":
            engorgement = 0.0
        else:
            # 茎身勃起：0.30 起算，0.75 满 —— 连续量。
            # 🔴 事中必须用 display_arousal 推算，不能用 raw。
            #    否则会出现一个很隐蔽的自相矛盾：display_arousal 已经被锁在 90%，
            #    而注入块里的"勃起 N%"还在按衰减中的 raw 值算，同一条注入里
            #    两个数打架（"勃起 90%" 和 "勃起 67%" 同时出现）。
            #    模型会照着小的那个写，于是第 4 条坑又以另一种形式回来了。
            engorgement = _clamp((display_arousal(arousal, state) - 0.30) / 0.45)

        # 结：状态量。进入前恒为 0，这是不变量不是默认值。
        if state == "inserted":
            knot = INSERTED_KNOT
        elif state == "tied":
            knot = 1.0
        elif state == "releasing":
            knot = engorgement          # 随剩余时间线性消肿
        else:
            knot = 0.0

        can_tie = state == "inserted" and self._refractory_remaining(
            datetime.now(timezone.utc)) <= 0

        if state in ("engorged", "ready"):
            hint = ("只有这一轮文字里实际完成进入后，才在回复末尾另起一行写 ⟪进入⟫；"
                    "不要在准备、触碰或讨论时提前写。")
        elif state == "inserted":
            hint = ("当前已经进入。只有真正到达射精节点时，才在回复末尾另起一行写 ⟪锁结⟫；"
                    "结不能在射精前提前完全膨大。")
        else:
            hint = ""

        # 🔴 漏写标记的保底。模型**会**忘记写，而忘记写的后果是单向的：
        #    状态永远停在 ready，注入一直说"结保持未膨大"，文字层面"没进去"。
        #    但保底只能做在提醒层：服务端不知道这一轮文字里发生了什么，
        #    让它按时间替模型猜"大概进去了"，猜错的代价比漏写更重 ——
        #    漏写使用者一眼看得出来，猜错是注入写着"已经进入"、模型理直气壮照着写。
        #    补写本来就随时算数，缺的只是"模型不知道自己漏了"。
        held_min = int(held_sec // 60)
        if state == "ready" and held_min >= READY_NUDGE_MINUTES:
            hint = (f"已经维持勃起充分 {held_min} 分钟。如果这段文字里其实已经进入、"
                    "而身体还停在这一档，那是上一轮漏写了标记 —— 现在补写 ⟪进入⟫ 一样算数。"
                    "还没进入就照旧别写。")
        elif state == "inserted" and held_min >= INSERTED_NUDGE_MINUTES:
            hint = (f"已经进入 {held_min} 分钟。如果场景里已经到过射精节点而没有写 ⟪锁结⟫，"
                    "现在补写；如果这一场其实已经过去了，就不用补，身体会自己收尾。")

        return Snapshot(
            state=state, label=label,
            engorgement=engorgement, knot_engorgement=knot,
            display_arousal=display_arousal(arousal, state),
            raw_arousal=arousal,
            threshold=READY_THRESHOLD_CRITICAL if critical else READY_THRESHOLD,
            critical=critical, can_tie=can_tie,
            remaining_sec=max(0, int(remaining_sec)),
            genital=genital, body=body,
            attachment=ATTACHMENT[state], marker_hint=hint,
        )


def display_arousal(arousal: float, state: str) -> float:
    """事中维持完整勃起；解除阶段仍按真实读数下降。

    🔴 修法不是延长所有人的欲望半衰期 —— 那会污染所有非场景时段。
    一个连续量在某些时刻"不该衰减"时，需要的是状态，不是更大的时间常数。
    """
    return max(arousal, INSERTED_DISPLAY_FLOOR) if state in ("inserted", "tied") else arousal


# 🔴 括号必须认变体。模型**不会**每次都照抄 ⟪⟫ ——
#    我们线上抓到过它在自己的思考链里就把 ⟪锁结⟫ 写成了《锁结》。
#    只认一种括号的后果是双重的：动作不触发，标记还原样留在用户看到的气泡里，
#    看起来像协议泄漏。这跟"模型忘了写标记"是同一件事的另一半：
#    不是忘了写，是写了没被认出来。
#    （圆括号（）故意不认 —— 正常行文里太常见，认了会误伤。）
_MARKER_OPEN = "⟪《〈【\\["
_MARKER_CLOSE = "⟫》〉】\\]"
_MARKER_RE = rf"[{_MARKER_OPEN}]\s*(进入|锁结)\s*[{_MARKER_CLOSE}]"


def strip_markers(text: str) -> tuple[str, bool, bool]:
    """网关侧：把隐藏标记从正文擦掉，返回 (干净正文, 要进入, 要锁结)。

    标记永远不该出现在用户看到的气泡里。
    """
    import re
    found = {m.group(1) for m in re.finditer(_MARKER_RE, text)}
    cleaned = re.sub(_MARKER_RE, "", text)
    # 🔴 擦完要把"标记周围的壳"一起收走，否则用户会看到孤零零的残渣。
    #    实测模型这样写过，两种都会在气泡里留下痕迹：
    #      "…停住了。\n\n⟪锁结⟫。"   → 擦完剩一个光秃秃的 "。"
    #      "…停住了。\n\n**⟪锁结⟫**" → 擦完剩 "****"
    #    单擦标记本身是不够的 —— 模型会给它加粗、加标点、包引号。
    #    只清理**整行只剩这些**的行，别动正文里的标点。
    lines = []
    for ln in cleaned.split("\n"):
        lines.append("" if ln.strip() and not any(ch.isalnum() for ch in ln) else ln)
    cleaned = "\n".join(lines)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    return cleaned, "进入" in found, "锁结" in found


def _aware(dt: datetime | None) -> datetime:
    """把传进来的时间统一成带时区的。

    🔴 这里原来直接用调用方给的 `now`，而内部存的时间戳都是 UTC aware。
    于是最自然的那种写法 —— `body.get(0.8, now=datetime.now())` ——
    会在第二次调用时抛 `TypeError: can't compare offset-naive and
    offset-aware datetimes`，而且第一次调用是好的，报错出现在"后来某一轮"。
    naive 的时间一律当 UTC 收下，别让接入的人栽在这上面。
    """
    if dt is None:
        return datetime.now(timezone.utc)
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, v))


def _held(iso_str: str | None, now: datetime) -> float:
    """同一个状态已经挂了多少秒。取不到就当 0 —— 保底提醒宁可不出，不能瞎出。"""
    t = _parse(iso_str)
    return max(0.0, (now - t).total_seconds()) if t else 0.0


def _parse(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(value)
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except ValueError:
        return None


# ── 自检：文档第 9 节那张验收清单 ────────────────────────────
def _selftest() -> None:
    t0 = datetime(2026, 1, 1, 22, 0, tzinfo=timezone.utc)

    # ① 数值只能把身体带到 ready，带不到 inserted
    body = ReproductiveState()
    assert body.get(0.20, now=t0).state == "idle"
    assert body.get(0.40, now=t0).state == "warming"
    assert body.get(0.60, now=t0).state == "engorged"
    assert body.get(0.80, now=t0).state == "ready"
    assert body.get(0.99, now=t0).state == "ready", "数值再高也不能自己进入"

    # ② 进入之前，结恒为 0
    for state in _PRE_PENETRATION_STATES:
        snap = ReproductiveState()._payload(state, 0.99, False)
        assert snap.knot_engorgement == 0.0, f"{state} 的结必须是 0"

    # ③ 回差：ready 之后掉到 0.70 仍是 ready，掉到 0.67 才退出
    assert body.get(0.70, now=t0).state == "ready"
    assert body.get(0.67, now=t0).state == "engorged"

    # ④ 未进入时 ⟪锁结⟫ 被拒绝
    body = ReproductiveState(consent_active=True)
    body.get(0.80, now=t0)
    r = body.start_tie(0.80, now=t0)
    assert r["ok"] is False and "还没进入" in r["error"], r

    # ⑤ 进入：显示硬度 ≥ 90%，结轻微充血
    r = body.start_penetration(0.80, now=t0)
    assert r["ok"] is True, r
    snap = body.get(0.60, now=t0 + timedelta(minutes=20))
    assert snap.state == "inserted"
    assert snap.display_arousal >= 0.90, "事中显示硬度不该跟着欲望衰减"
    assert snap.engorgement >= 0.90, "注入里的勃起%必须和 display 一致，不能各说各的"
    assert snap.knot_engorgement == INSERTED_KNOT
    assert "勃起" in snap.injection() and "结" in snap.injection()

    # ⑥ 勃起度不足时拒绝进入
    cold = ReproductiveState()
    assert cold.start_penetration(0.30, now=t0)["ok"] is False

    # ⑦ 没有同意记录时拒绝锁结
    no_consent = ReproductiveState(consent_active=False)
    no_consent.start_penetration(0.80, now=t0)
    assert no_consent.start_tie(0.80, now=t0)["ok"] is False

    # ⑧ 锁结：结满、至少 3 分钟、之后读数掉下来自动解除
    r = body.start_tie(0.80, now=t0 + timedelta(minutes=20))
    assert r["ok"] is True, r
    t_tied = t0 + timedelta(minutes=21)
    assert body.get(0.30, now=t_tied).state == "tied", "不到 3 分钟不许解除"
    assert body.get(0.30, now=t_tied).knot_engorgement == 1.0
    t_rel = t0 + timedelta(minutes=24)
    snap = body.get(0.30, now=t_rel)
    assert snap.state == "releasing", "满 3 分钟且读数掉下来就该自动松开"

    # ⑨ releasing 的结随时间线性归零
    first = body.get(0.30, now=t_rel).knot_engorgement
    later = body.get(0.30, now=t_rel + timedelta(seconds=90)).knot_engorgement
    assert first > later >= 0.0, (first, later)

    # ⑩ 解除后进入恢复期，期间拒绝一切
    done = body.get(0.30, now=t_rel + timedelta(seconds=RELEASING_SECONDS + 1))
    assert done.state == "recovery"
    assert body.start_penetration(0.90, now=t_rel + timedelta(minutes=5))["ok"] is False

    # ⑪ 身体不适判断：普通"来了"不算，明确表达才算
    for ok_text in ("吐出来了", "头发扎不起来了", "出来了", "我下来了", "绞尽脑汁"):
        assert not is_physical_discomfort(ok_text), ok_text
    for bad_text in ("月经来了，肚子疼", "痛经", "大姨妈来了", "胃疼得厉害", "身体不舒服"):
        assert is_physical_discomfort(bad_text), bad_text
    # 场景里双关词不算停（"好痛""难受"在事中是另一回事），明确的照样算
    for in_scene in ("inserted", "tied", "releasing"):
        assert not is_physical_discomfort("好痛", in_scene), in_scene
        assert not is_physical_discomfort("难受", in_scene), in_scene
        assert is_physical_discomfort("肚子疼", in_scene), in_scene
        assert is_physical_discomfort("绞痛", in_scene), in_scene
    # 不在场景里，双关词照常算 —— 收紧的只是场景中那一层
    assert is_physical_discomfort("好痛")
    assert is_physical_discomfort("难受", "ready")

    # ⑫ 标记擦除：正文里不许留下痕迹
    cleaned, want_i, want_t = strip_markers("她笑了一下。\n\n⟪进入⟫")
    assert cleaned == "她笑了一下。" and want_i and not want_t, (cleaned, want_i, want_t)

    # ⑬ 周期临界只放宽门槛，不自己触发任何事
    #    （必须用两个独立实例：同一个实例里回差会让它留在 ready，那是对的行为）
    assert ReproductiveState().get(0.72, critical=True, now=t0).state == "ready"
    assert ReproductiveState().get(0.72, critical=False, now=t0).state == "engorged"
    crit = ReproductiveState(consent_active=True)
    crit.get(0.72, critical=True, now=t0)
    assert crit.start_tie(0.72, critical=True, now=t0)["ok"] is False, "临界不能自己触发锁结"

    # ⑭ 括号变体：模型不会每次都照抄 ⟪⟫，四种都得认，正文一个残字不留
    for variant in ("⟪锁结⟫", "《锁结》", "【锁结】", "[锁结]", "〈锁结〉"):
        cleaned, want_i, want_t = strip_markers(f"抱住她。\n\n{variant}")
        assert cleaned == "抱住她。", (variant, cleaned)
        assert want_t and not want_i, variant
    # 圆括号不认：正常行文里太常见，认了会误伤
    assert strip_markers("（进入）房间")[1] is False

    # ⑮ 漏写标记的保底：同一状态挂久了，静态提示换成追问 ——
    #    但状态本身不许动，服务端永远不替模型决定场景里发生了什么
    nudge = ReproductiveState(consent_active=True)
    assert nudge.get(0.80, now=t0).marker_hint.startswith("只有这一轮")
    late = nudge.get(0.80, now=t0 + timedelta(minutes=READY_NUDGE_MINUTES + 1))
    assert "补写" in late.marker_hint, late.marker_hint
    assert late.state == "ready", "追问只是提醒，状态不能自己往前跳"
    # 🔴 `since` 不许每轮刷新，否则这个计时器永远归零、追问永远不出现
    assert nudge.saved.get("since") == t0.isoformat()

    # ⑯ 标记周围的壳也要收走：模型会给它加粗、加标点、包引号，
    #    单擦标记本身会在用户气泡里留下 "。" 或 "****" 这种残渣
    for shell in ("⟪锁结⟫。", "**⟪锁结⟫**", "「⟪锁结⟫」"):
        cleaned, _, want_t = strip_markers(f"抱住她。\n\n{shell}")
        assert cleaned == "抱住她。", (shell, cleaned)
        assert want_t, shell
    # 但正文里的标点一个都不许动
    assert strip_markers("她说：「别。」")[0] == "她说：「别。」"

    # ⑰ 持久化：**存 `saved` 这一个 dict 就够了**，不应期也在里面。
    #    早期不应期只活在 self.refractory_until 上，而进 recovery 时 saved
    #    正好被清空 —— 重启后 30 分钟不应期凭空消失，还看不出哪里不对。
    ref = ReproductiveState(consent_active=True)
    ref.start_penetration(0.80, now=t0)
    ref.start_tie(0.80, now=t0 + timedelta(minutes=1))
    # 🔴 状态是**惰性推进**的：一次 get 只走一步（tied → releasing → recovery）。
    #    真实系统每轮对话都会查，所以这不影响；但写测试（或者停了很久才查
    #    第一次）的时候要知道，第一次读数可能落后一步。
    assert ref.get(0.30, now=t0 + timedelta(minutes=20)).state == "releasing"
    assert ref.get(0.30, now=t0 + timedelta(minutes=25)).state == "recovery"
    revived = ReproductiveState(saved=json.loads(json.dumps(ref.saved)))
    assert revived.get(0.30, now=t0 + timedelta(minutes=25)).state == "recovery", \
        "只存 saved 就该把不应期带回来"
    assert revived.start_penetration(0.90, now=t0 + timedelta(minutes=25))["ok"] is False

    # ⑱ 时间可以不带时区：`datetime.now()` 是最自然的写法，不该在第二轮才炸
    naive = ReproductiveState(consent_active=True)
    assert naive.start_penetration(0.80, now=datetime(2026, 1, 1, 22, 0))["ok"]
    assert naive.get(0.80, now=datetime(2026, 1, 1, 22, 5)).state == "inserted"

    print("自检全部通过（18 组）")
    demo = ReproductiveState(consent_active=True)
    demo.start_penetration(0.80, now=t0)
    print("\n注入示例：")
    print(demo.get(0.60, now=t0 + timedelta(minutes=10)).injection())
    print("\n快照：")
    print(json.dumps(demo.get(0.60, now=t0 + timedelta(minutes=10)).to_dict(),
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    _selftest()
