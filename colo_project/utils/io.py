"""Result persistence: metrics as JSON, arrays as .npz, figures as PNG."""

import json
import subprocess
from pathlib import Path

import numpy as np


def result_dir(root: Path, scenario_name: str, seed: int) -> Path:
    """Allocate a fresh `<root>/<scenario>_seed<seed>/run_NNNN/`.

    Every invocation gets its own directory, so a rerun can no longer clobber
    the one before it -- which it silently did when this returned the scenario
    directory itself. That also settles the case of two runs whose only
    difference is an algorithm knob: `seed` here is the *scenario* seed, so
    `run_nbp --seed 0` and `--seed 7` used to land on the same files.

    The index is one past the highest existing `run_NNNN`, not a count, so
    deleting a run does not hand its number out again. `figures/` is created
    eagerly; the scenario directory is `.parent`, and holds the figures that
    depend only on the scenario.
    """
    parent = Path(root) / f"{scenario_name}_seed{seed}"
    parent.mkdir(parents=True, exist_ok=True)
    used = [int(p.name[4:]) for p in parent.glob("run_*")
            if p.is_dir() and p.name[4:].isdigit()]
    nxt = max(used, default=0) + 1
    for i in range(nxt, nxt + 100):
        out = parent / f"run_{i:04d}"
        try:
            # No exist_ok: mkdir failing *is* how a lost race is detected.
            out.mkdir()
        except FileExistsError:
            continue
        (out / "figures").mkdir()
        return out
    raise RuntimeError(f"no free run directory under {parent}")


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


def save_fig(fig, path: Path, dpi: int = 120) -> Path:
    """Write `fig` to `path`.

    Deliberately does not close: `--show` needs the figure alive afterwards,
    and the scripts already close everything once at the end.
    """
    path = Path(path)
    fig.savefig(path, dpi=dpi)
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
