"""A tiny, dependency-free embodied-state reference for companion AIs."""

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

        organs = self._organs()
        ears = self._ears(event)
        tail = self._tail()
        paws = self._paws(event)
        throat = self._throat()

        return {
            "chemistry": vars(self.chemistry).copy(),
            "organs": organs,
            "ears": ears,
            "tail": tail,
            "paws": paws,
            "throat": throat,
            "prompt_hint": self._hint(organs, ears, tail, paws, throat),
        }

    def _organs(self) -> dict[str, str]:
        c = self.chemistry
        stomach = "发紧" if c.cortisol_like > 0.68 else "空落" if c.dopamine_like < 0.25 else "安稳"
        chest = "发热" if c.oxytocin_like > 0.68 else "绷着" if c.adrenaline_like > 0.7 else "平缓"
        breath = "急" if c.adrenaline_like > 0.72 else "浅" if c.cortisol_like > 0.65 else "匀"
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
    print(state["prompt_hint"])
    print("self-check: ok")


if __name__ == "__main__":
    _self_check()
