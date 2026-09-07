"""Result persistence: metrics as JSON, arrays as .npz, figures as PNG."""

import json
import subprocess
from pathlib import Path

import numpy as np


def result_dir(root: Path, scenario_name: str, seed: int) -> Path:
    """`<root>/<scenario>_seed<seed>/`, created if missing."""
    out = Path(root) / f"{scenario_name}_seed{seed}"
    out.mkdir(parents=True, exist_ok=True)
    return out


def _jsonable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"not JSON serializable: {type(obj)}")


def save_json(path: Path, obj: dict) -> Path:
    path = Path(path)
    path.write_text(json.dumps(obj, indent=2, default=_jsonable))
    return path


def load_json(path: Path) -> dict:
    return json.loads(Path(path).read_text())


def save_arrays(path: Path, **arrays) -> Path:
    path = Path(path)
    np.savez_compressed(path, **arrays)
    return path


def git_sha() -> str:
    """
    Short SHA of HEAD, or 'unknown' outside a git checkout.

    Resolved against the repo this package lives in, not the caller's cwd, so
    results stay traceable when a script is run from elsewhere.
    """
    repo = Path(__file__).resolve().parents[2]
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=repo, capture_output=True, text=True, check=True,
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return "unknown"
