from __future__ import annotations

from enum import Enum
from typing import List, Tuple, Optional


class PairDirection(str, Enum):


    A_TO_B = "a_to_b"
    B_TO_A = "b_to_a"
    BOTH = "both"


STAGE_TO_CONFIG_KEY: dict[str, str] = {

    "cloth_fit_reshaping": "11c_cloth_fit_reshaping",
}


def _parse_pair_direction(value: Optional[object]) -> PairDirection:


    if value is None:
        return PairDirection.BOTH


    if isinstance(value, PairDirection):
        return value

    s = str(value).strip().lower()
    if s in {"both", "bi", "bidirectional", "bi-directional", "a_to_b_and_b_to_a"}:
        return PairDirection.BOTH
    if s in {"a_to_b", "a->b", "ab", "forward"}:
        return PairDirection.A_TO_B
    if s in {"b_to_a", "b->a", "ba", "reverse"}:
        return PairDirection.B_TO_A

    raise ValueError(
        f"Invalid pair_direction='{value}'. Expected one of: "
        f"{PairDirection.A_TO_B.value}, {PairDirection.B_TO_A.value}, {PairDirection.BOTH.value}"
    )


def get_pair_directions(
    *,
    stage_name: str,
    config: dict,
    subjects: List[str],
) -> List[Tuple[str, str]]:


    if len(subjects) != 2:
        raise ValueError(
            f"get_pair_directions expects exactly 2 subjects, got {len(subjects)}: {subjects}"
        )

    stage_cfg_key = STAGE_TO_CONFIG_KEY.get(stage_name)
    stage_cfg = (
        config.get("pipeline_stages", {}).get(stage_cfg_key, {}) if stage_cfg_key else {}
    )
    stage_override = stage_cfg.get("pair_direction")
    global_default = config.get("execution", {}).get("pair_direction")

    mode = _parse_pair_direction(stage_override if stage_override is not None else global_default)

    a, b = subjects[0], subjects[1]
    if mode == PairDirection.A_TO_B:
        return [(a, b)]
    if mode == PairDirection.B_TO_A:
        return [(b, a)]
    return [(a, b), (b, a)]
