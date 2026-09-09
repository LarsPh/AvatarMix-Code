from __future__ import annotations

from typing import Iterable, Optional


def fallback_order_for_target(target_label: str) -> list[str]:

    target = str(target_label)
    if target in ("left_arm", "right_arm"):
        return ["arms_both", "torso_non_neck", "legs_both", "neck", "hands_both"]
    if target in ("left_leg", "right_leg"):
        return ["legs_both", "torso_non_neck", "arms_both", "neck", "hands_both"]
    if target in ("torso_skin", "torso_non_neck"):
        return ["torso_non_neck", "arms_both", "legs_both", "neck", "hands_both"]
    if target == "hands":

        return ["hands_both", "arms_both", "torso_non_neck", "legs_both", "neck"]
    return [target, "torso_non_neck", "arms_both", "legs_both", "neck"]


def resolve_source_key_for_target(target_label: str, available_sources: Iterable[str]) -> Optional[str]:

    target = str(target_label)
    avail = set(available_sources)

    order = fallback_order_for_target(target)

    for k in order:
        if k in avail:
            return k
    return None
