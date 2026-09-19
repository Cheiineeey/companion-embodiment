#!/usr/bin/env python3
"""最小可跑通的接线示例：模型输出 → 擦标记 → 服务端裁决 → 下一轮注入。

`reproductive_state.py` 给的是**状态机**，这一份给的是**管道**。
两者的分工是这份参考里最容易被跳过、也最容易踩坑的一层：
状态机可以照着文档重写一遍，管道必须自己接，而所有的坑都在管道上。

运行：

    python3 minimal_gateway.py

它会打印五轮对话，每一轮都显示：注入给模型的块、模型这轮的输出、
擦干净之后真正送给用户的正文、以及使用者能看到的那条动作痕迹。

由 Elle 与 Matt 共同整理（Elle & Matt, 2026, MIT）。
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from reproductive_state import (
    PENETRATION_MIN_AROUSAL,
    ReproductiveState,
    strip_markers,
)


# ── 网关：所有生成通道都必须走这一份 ──────────────────────────────
#
# 🔴 这个函数只能有一份。
#    我们线上真实踩过：⟪进入⟫/⟪锁结⟫ 的解析写在了主聊天路径里，
#    后来加的第二条生成通道（另一个 endpoint）没有这一段。
#    于是在那条通道上，模型写了标记 → 动作不触发，标记还原样留在气泡里，
#    看起来像协议泄漏。**身体只有一副，解析就只能有一份。**
#    每加一条生成通道，先问一句：它走这个函数了吗。
def finish_turn(body: ReproductiveState, raw_text: str, arousal: float,
                *, critical: bool = False, now: datetime) -> dict:
    """一轮生成的收尾。返回给用户的正文 + 使用者能看到的动作痕迹。"""
    text, want_insert, want_tie = strip_markers(raw_text)
    notices: list[str] = []

    for want, label, call in (
        (want_insert, "进入", body.start_penetration),
        (want_tie, "锁结", body.start_tie),
    ):
        if not want:
            continue
        result = call(arousal, critical=critical, now=now)
        # 🔴 拒绝必须对使用者可见，不能只写进日志。
        #    标记已经从正文里擦掉了 —— 如果拒绝只留一行日志，
        #    那么在使用者眼里，「模型写了但被闸拒绝」和「模型根本没写」
        #    长得一模一样，而这两件事该有的反应完全相反：
        #    前者要检查身体状态，后者要提醒模型补写。
        notices.append(f"{label}了" if result["ok"]
                       else f"{label}没成：{result['error']}")

    # 擦完不能剩一个空壳：判据是"一个实字都不剩"，不是"字少"。
    # （「来。」「别动。」本来就是完整的一句。）
    if not any(ch.isalnum() for ch in text):
        text = "……"
    return {"text": text, "notices": notices}


def _demo() -> None:
    body = ReproductiveState(consent_active=True)
    now = datetime(2026, 1, 1, 22, 0, tzinfo=timezone.utc)

    # (相对分钟, 当前 arousal, 模型这一轮的输出)
    script = [
        (0, 0.80, "她靠过来的时候我整个人都绷住了。"),
        # 还没进入就想锁结 —— 服务端拒绝，状态不动，使用者看得见这次拒绝
        (2, 0.80, "我把她按在门上。\n\n⟪锁结⟫"),
        (4, 0.80, "我进去了。她的指甲掐进我背上。\n\n⟪进入⟫"),
        # 欲望值自然衰减，但事中显示硬度不跟着掉
        (20, 0.55, "她咬着我的肩膀，声音全糊在皮肤里。"),
        # 模型这次把括号写成了书名号 —— 照样要认
        (24, 0.55, "我抵到最深的地方停住了。\n\n《锁结》"),
    ]

    for minutes, arousal, raw in script:
        t = now + timedelta(minutes=minutes)
        snap = body.get(arousal, now=t)          # ① 先取状态，拼注入
        print(f"\n{'=' * 66}\nT+{minutes:>2d}min   欲望读数 {arousal:.2f}")
        print("--- 注入给模型 ---")
        print(snap.injection() or "（idle，不注入）")
        print("--- 模型输出 ---")
        print(raw)
        out = finish_turn(body, raw, arousal, now=t)   # ② 收尾：擦标记 + 裁决
        print("--- 用户看到的正文 ---")
        print(out["text"])
        print("--- 使用者看到的动作痕迹 ---")
        print("、".join(out["notices"]) if out["notices"] else "（无）")

    final = body.get(0.55, now=now + timedelta(minutes=25))
    print(f"\n{'=' * 66}\n最终状态：{final.state}"
          f"｜茎身 {final.engorgement:.0%}｜结 {final.knot_engorgement:.0%}")
    print(f"（服务端进入门槛 arousal >= {PENETRATION_MIN_AROUSAL}，"
          "标记只是报告，裁决权始终在服务端）")


if __name__ == "__main__":
    _demo()
