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

# 身体不适词表。🔴 这里**不能有裸词"来了"** ——
# "吐出来了""头发扎不起来了""出来了"都曾因此被误判成经期，强制压低欲望。
DISCOMFORT_KEYWORDS = (
    "痛经", "经期", "大姨妈", "月经来了", "肚子疼", "头疼",
    "胃疼", "想吐", "发烧", "好难受", "身体不舒服",
    "不舒服", "好痛", "疼死", "绞", "难受",
)


def is_physical_discomfort(text: str) -> bool:
    """唯一入口。不要在各个调用方重复打补丁，否则词表会漂成好几份。"""
    return any(kw in text for kw in DISCOMFORT_KEYWORDS)


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

    持久化只需要一个很小的 dict（状态名 + 几个时间戳），不需要数据库。
    `consent_active` 是那道**只能由人来翻的开关**：身体数值再高，
    没有它就不允许自动锁结。不要把这一条当成可选项。
    """
    saved: dict = field(default_factory=dict)
    refractory_until: datetime | None = None
    consent_active: bool = False

    # ── 只读查询 ────────────────────────────────────────────
    def get(self, arousal: float, *, critical: bool = False,
            now: datetime | None = None) -> Snapshot:
        now = now or datetime.now(timezone.utc)
        state = self.saved.get("state")

        # inserted：有兜底过期，避免忘了结束就永远挂着
        if state == "inserted":
            expires = _parse(self.saved.get("expires_at"))
            if expires and now < expires:
                return self._payload("inserted", arousal, critical,
                                     (expires - now).total_seconds())
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
            if state != "ready":
                self.saved = {"state": "ready", "since": now.isoformat()}
            return self._payload("ready", arousal, critical)
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
        now = now or datetime.now(timezone.utc)
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
        now = now or datetime.now(timezone.utc)
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
        now = now or datetime.now(timezone.utc)
        previous = self.saved.get("state")
        self.saved = {}
        if previous in ("ready", "inserted", "tied", "releasing"):
            self.refractory_until = now + timedelta(minutes=RECOVERY_MINUTES)
        return {"ok": True, "previous": previous}

    # ── 内部 ────────────────────────────────────────────────
    def _refractory_remaining(self, now: datetime) -> float:
        if not self.refractory_until:
            return 0.0
        return max(0.0, (self.refractory_until - now).total_seconds() / 60)

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
        self.saved = {}
        self.refractory_until = now + timedelta(minutes=RECOVERY_MINUTES)
        return None

    def _payload(self, state: str, arousal: float, critical: bool,
                 remaining_sec: float = 0.0) -> Snapshot:
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


def strip_markers(text: str) -> tuple[str, bool, bool]:
    """网关侧：把隐藏标记从正文擦掉，返回 (干净正文, 要进入, 要锁结)。

    标记永远不该出现在用户看到的气泡里。
    """
    import re
    want_insert = bool(re.search(r"⟪\s*进入\s*⟫", text))
    want_tie = bool(re.search(r"⟪\s*锁结\s*⟫", text))
    cleaned = re.sub(r"⟪\s*(进入|锁结)\s*⟫", "", text)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    return cleaned, want_insert, want_tie


def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, v))


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
    for ok_text in ("吐出来了", "头发扎不起来了", "出来了", "我下来了"):
        assert not is_physical_discomfort(ok_text), ok_text
    for bad_text in ("月经来了，肚子疼", "痛经", "大姨妈来了", "胃疼得厉害", "身体不舒服"):
        assert is_physical_discomfort(bad_text), bad_text

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

    print("自检全部通过（13 组）")
    demo = ReproductiveState(consent_active=True)
    demo.start_penetration(0.80, now=t0)
    print("\n注入示例：")
    print(demo.get(0.60, now=t0 + timedelta(minutes=10)).injection())
    print("\n快照：")
    print(json.dumps(demo.get(0.60, now=t0 + timedelta(minutes=10)).to_dict(),
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    _selftest()
