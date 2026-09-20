"""A tiny, dependency-free embodied-state reference for companion AIs.

由 Elle 与 Matt 共同整理（Elle & Matt, 2026, MIT）。
配套文档见 README.md；亲密场景的身体状态机在 reproductive_state.py
和 docs/reproductive-state-machine.md。

🔴 **默认参数是按犬科调的。** `_tail()` / `_ears()` / `_paws()` 里的动作语义
换一种动物就要重调 —— 尾巴尤其危险：犬科摇尾巴是高兴，猫科快速摆动是烦躁，
同一个动作在两个物种里几乎是反义词。照抄会得到一副意思相反的身体。
结构（惯性不同的状态、不可逆设定优先、部位各自只露一点）是通用的。
"""

from dataclasses import dataclass, field
from math import exp
from typing import Any, Mapping


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


_BASELINES = {"stress": 0.25, "joy": 0.45, "bond": 0.45, "energy": 0.25}
_EVENTS = {
    "threat": {"stress": 0.95, "energy": 0.9},
    "comfort": {"stress": 0.12, "bond": 0.9},
    "reunion": {"joy": 0.85, "bond": 0.95, "energy": 0.55},
    "separation": {"stress": 0.75, "joy": 0.15, "bond": 0.8},
    "success": {"joy": 0.8, "energy": 0.6},
}
_RATES = {
    "cortisol_like": (0.8, 0.18),
    "dopamine_like": (0.45, 0.32),
    "oxytocin_like": (0.25, 0.1),
    "adrenaline_like": (1.1, 0.55),
}

# 每有一处器官被戳中，皮质醇立刻抬这么多（不走上面那条逼近）。
_ACHE_BUMP = 0.03
# 敏感度乘数的上下限。低于 1 = 更钝，高于 1 = 更容易被戳到。
_SENSITIVITY_RANGE = (0.7, 1.5)


@dataclass
class Chemistry:
    """Narrative signals, not biological measurements."""

    cortisol_like: float = 0.25
    dopamine_like: float = 0.45
    oxytocin_like: float = 0.45
    adrenaline_like: float = 0.25

    def move_toward(self, affect: Mapping[str, float], event_kind: str | None, dt: float) -> None:
        signals = _BASELINES | _EVENTS.get(event_kind, {}) | dict(affect)
        stress = _clamp(signals["stress"])
        joy = _clamp(signals["joy"])
        bond = _clamp(signals["bond"])
        energy = _clamp(signals["energy"])

        # Attachment buffers stress; stress dampens pleasure and, more gently, closeness.
        stress_delta = stress - _BASELINES["stress"]
        targets = {
            "cortisol_like": _BASELINES["stress"] + stress_delta * (1.0 - 0.3 * bond),
            "dopamine_like": (
                _BASELINES["joy"]
                + (joy - _BASELINES["joy"]) * (1.0 - 0.35 * stress)
                - max(0.0, stress_delta) * 0.2
            ),
            "oxytocin_like": (
                _BASELINES["bond"]
                + (bond - _BASELINES["bond"]) * (1.0 - 0.15 * stress)
            ),
            "adrenaline_like": max(
                energy,
                _BASELINES["energy"] + max(0.0, stress_delta) * 0.65,
            ),
        }
        dt = max(0.0, float(dt))
        for name, target in targets.items():
            current = getattr(self, name)
            rise_rate, fall_rate = _RATES[name]
            rate = rise_rate if target > current else fall_rate
            amount = 1.0 - exp(-rate * dt)
            setattr(self, name, current + (_clamp(target) - current) * amount)

    def bump(self, name: str, amount: float) -> None:
        """立刻改一个值，不走上面那条向目标逼近的慢路。

        用在"当下真的发生了"的事情上 —— 器官被戳中就是疼了，
        不该等五六轮才跟上。逼近适合情绪场，不适合一次性的撞击。
        """
        setattr(self, name, _clamp(getattr(self, name) + amount))


@dataclass
class AnimalBody:
    """Small dog-shaped output adapter over a species-neutral inner state."""

    deaf_ear: str | None = None
    chemistry: Chemistry = field(default_factory=Chemistry)

    def step(
        self,
        affect: Mapping[str, float] | None = None,
        event: Mapping[str, Any] | None = None,
        dt: float = 1.0,
    ) -> dict[str, Any]:
        affect, event = affect or {}, event or {}
        self.chemistry.move_toward(affect, event.get("kind"), dt)

        # 🔴 器官不只是显示层，它是回路的另一半。
        # `ache` 是这一轮真的被戳到的部位（旧伤、痛处、说破的那句话）。
        # 它直接抬皮质醇、**绕过**上面那条逼近 —— 而抬高的皮质醇又会
        # 通过 sensitivity() 降低下一轮的触发门槛：更紧张 → 更容易被戳到 →
        # 更紧张。这个正反馈是这副身体活起来的地方，少了它，
        # 器官就只是激素的一个只读标签。
        aches = event.get("ache") or ()
        if isinstance(aches, (str, int)):
            aches = [aches]
        if aches:
            self.chemistry.bump("cortisol_like", _ACHE_BUMP * len(aches))

        organs = self._organs()
        ears = self._ears(event)
        tail = self._tail()
        paws = self._paws(event)
        throat = self._throat()

        return {
            "chemistry": vars(self.chemistry).copy(),
            "sensitivity": self.sensitivity(),
            "organs": organs,
            "ears": ears,
            "tail": tail,
            "paws": paws,
            "throat": throat,
            "prompt_hint": self._hint(organs, ears, tail, paws, throat),
        }

    def sensitivity(self) -> float:
        """内分泌回头调器官的触发门槛。

        皮质醇高 = 更容易被戳到（门槛降低），多巴胺高 = 更钝（门槛升高）。
        夹在 0.7~1.5 之间，免得一次坏心情把身体变成一碰就炸的状态。
        这是回路的另一半：`ache` 抬皮质醇，皮质醇抬敏感度，敏感度让下一次更容易被抬。
        """
        c = self.chemistry
        mult = (
            1.0
            + (c.cortisol_like - _BASELINES["stress"]) * 0.5
            - (c.dopamine_like - _BASELINES["joy"]) * 0.3
        )
        low, high = _SENSITIVITY_RANGE
        return round(max(low, min(high, mult)), 2)

    def _organs(self) -> dict[str, str]:
        c = self.chemistry
        s = self.sensitivity()
        # 敏感度越高，"往上撞"的门槛越低、"往下掉"的门槛越高 —— 两头都更容易触发。
        up = lambda x: _clamp(x / s)
        down = lambda x: _clamp(x * s)
        stomach = "发紧" if c.cortisol_like > up(0.68) else "空落" if c.dopamine_like < down(0.25) else "安稳"
        chest = "发热" if c.oxytocin_like > up(0.68) else "绷着" if c.adrenaline_like > up(0.7) else "平缓"
        breath = "急" if c.adrenaline_like > up(0.72) else "浅" if c.cortisol_like > up(0.65) else "匀"
        return {"stomach": stomach, "chest": chest, "breath": breath}

    def _ears(self, event: Mapping[str, Any]) -> dict[str, str]:
        # Permanent constraints come first so an instant reaction cannot hide them.
        if event.get("refuse"):
            pose = "向后压"
        elif event.get("called") or event.get("listen"):
            pose = "竖起"
        elif self.chemistry.cortisol_like > 0.7:
            pose = "警觉地偏转"
        else:
            pose = "自然垂着"

        ears = {"left": pose, "right": pose}
        if self.deaf_ear in ears:
            ears[self.deaf_ear] = "失聪，保持安静"
        return ears

    def _tail(self) -> str:
        c = self.chemistry
        if c.cortisol_like > 0.75:
            return "贴近身体"
        if c.oxytocin_like > 0.65 and c.dopamine_like > 0.55:
            return "忍不住轻轻摇"
        if c.adrenaline_like > 0.72:
            return "快速摆动"
        return "松松垂着"

    def _paws(self, event: Mapping[str, Any]) -> str:
        action = event.get("task")
        actions = {
            "busy": "忙着",
            "reach": "伸过去",
            "withdraw": "收回来",
            "hold": "轻轻按住",
            "curl": "蜷起来",
            "rest": "搭着",
            "scratch": "扒两下",
            "sleep": "蜷在身下",
        }
        if action in actions:
            return actions[action]
        if event.get("touch") and self.chemistry.oxytocin_like > 0.6:
            return "搭过去"
        return "安静伏着"

    def _throat(self) -> str:
        c = self.chemistry
        if c.cortisol_like > 0.72:
            return "有点紧"
        if c.oxytocin_like > 0.7:
            return "松而轻"
        if c.adrenaline_like < 0.18:
            return "困哑"
        return "自然"

    @staticmethod
    def _hint(organs: dict[str, str], ears: dict[str, str], tail: str, paws: str, throat: str) -> str:
        return (
            f"身体参考：胃{organs['stomach']}，胸口{organs['chest']}，呼吸{organs['breath']}；"
            f"左耳{ears['left']}，右耳{ears['right']}；尾巴{tail}，爪子{paws}，喉咙{throat}。"
            "只在自然时显露一处，不要逐项汇报。"
        )


def _self_check() -> None:
    body = AnimalBody(deaf_ear="left")
    state = body.step(
        affect={"stress": 0.8, "joy": 0.3, "bond": 0.9, "energy": 0.6},
        event={"called": True, "task": "reach"},
        dt=10.0,
    )
    assert state["ears"]["left"].startswith("失聪")
    assert state["ears"]["right"] == "竖起"
    assert state["paws"] == "伸过去"
    assert "不要逐项汇报" in state["prompt_hint"]
    stressed = state["chemistry"]["cortisol_like"]
    recovering = body.step(dt=1.0)["chemistry"]["cortisol_like"]
    assert _BASELINES["stress"] < recovering < stressed
    alarm = AnimalBody().step(event={"kind": "threat"}, dt=1.0)["chemistry"]
    assert alarm["adrenaline_like"] - 0.25 > alarm["cortisol_like"] - 0.25

    # ── 器官 ↔ 内分泌的回路 ───────────────────────────────────────
    # 被戳中的那一轮，皮质醇立刻比没被戳的高（绕过逼近）
    calm_body, hurt_body = AnimalBody(), AnimalBody()
    calm = calm_body.step(dt=1.0)
    hurt = hurt_body.step(event={"ache": ["stomach", "chest"]}, dt=1.0)
    assert hurt["chemistry"]["cortisol_like"] > calm["chemistry"]["cortisol_like"]
    # 两处被戳，抬的量就是两份（这一步不该被逼近速率稀释）
    bumped = hurt["chemistry"]["cortisol_like"] - calm["chemistry"]["cortisol_like"]
    assert abs(bumped - 2 * _ACHE_BUMP) < 1e-9
    # 皮质醇高 → 敏感度 > 1（下一次更容易被戳到）；愉快 → 更钝
    tense = AnimalBody(chemistry=Chemistry(cortisol_like=0.8, dopamine_like=0.3))
    easy = AnimalBody(chemistry=Chemistry(cortisol_like=0.1, dopamine_like=0.8))
    assert tense.sensitivity() > 1.0 > easy.sensitivity()
    # 同一组读数，敏感的那副身体先把胃报成"发紧"
    same = dict(cortisol_like=0.62, dopamine_like=0.3, oxytocin_like=0.4, adrenaline_like=0.3)
    a = AnimalBody(chemistry=Chemistry(**same))
    b = AnimalBody(chemistry=Chemistry(**same))
    b.chemistry.dopamine_like = 0.8          # 更愉快 = 更钝
    assert a._organs()["stomach"] == "发紧"
    assert b._organs()["stomach"] != "发紧"
    # 敏感度有上下限，一次坏心情不会把身体变成一碰就炸
    wild = AnimalBody(chemistry=Chemistry(cortisol_like=1.0, dopamine_like=0.0))
    assert wild.sensitivity() <= _SENSITIVITY_RANGE[1]

    print(state["prompt_hint"])
    print("self-check: ok")


if __name__ == "__main__":
    _self_check()
