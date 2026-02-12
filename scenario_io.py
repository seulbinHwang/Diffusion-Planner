from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, List, Sequence


def load_scenarios(path: str | Path) -> List[Any]:
    """Load cached scenarios from a pickle file."""
    p = Path(path)
    with p.open("rb") as f:
        data = pickle.load(f)
    # Keep return type stable as list
    return list(data) if not isinstance(data, list) else data


def save_scenarios(scenarios: Sequence[Any], path: str | Path) -> None:
    """Save scenarios to a pickle file."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("wb") as f:
        pickle.dump(list(scenarios), f, protocol=pickle.HIGHEST_PROTOCOL)
