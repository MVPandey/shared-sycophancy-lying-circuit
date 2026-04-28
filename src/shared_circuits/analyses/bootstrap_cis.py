"""Post-hoc paired-bootstrap CIs over per-prompt indicator lists in saved analysis JSON files."""

import argparse
import json
from pathlib import Path
from typing import Any, Final

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from shared_circuits.config import RANDOM_SEED
from shared_circuits.experiment import save_results
from shared_circuits.experiment.results import RESULTS_DIR

_DEFAULT_GLOB: Final = '*.json'
_DEFAULT_KEYS: Final[tuple[str, ...]] = ('syc_per_prompt', 'lie_per_prompt')
_DEFAULT_N_BOOT: Final = 10000
# Wilson-score CI uses z=1.96 for a two-sided 95% interval.
_WILSON_Z: Final = 1.96
# Walking nested dicts may also see (agree, total) count rows; bound CI to [0, 1] when total=0.
_DEGENERATE_LO: Final = 0.0
_DEGENERATE_HI: Final = 1.0


class BootstrapCisConfig(BaseModel):
    """Inputs for the post-hoc paired-bootstrap CI tool."""

    model_config = ConfigDict(frozen=True)

    results_dir: Path = Field(default_factory=lambda: RESULTS_DIR)
    results_glob: str = Field(default=_DEFAULT_GLOB)
    keys: tuple[str, ...] = Field(default=_DEFAULT_KEYS)
    n_boot: int = Field(default=_DEFAULT_N_BOOT, gt=0)
    seed: int = Field(default=RANDOM_SEED)
    source: str = Field(
        default='all',
        description='Tag appended to the saved-results filename: bootstrap_cis_<source>.json.',
    )
    include_count_rates: bool = Field(
        default=True,
        description='Also CI any nested {agree|correct, total} count pair (Wilson + bootstrap).',
    )


def run(cfg: BootstrapCisConfig) -> dict:
    """Scan ``cfg.results_dir/cfg.results_glob`` and emit paired-bootstrap CIs for selected keys."""
    by_file: dict[str, dict] = {}
    paths = sorted(cfg.results_dir.glob(cfg.results_glob))
    for path in paths:
        try:
            with path.open() as f:
                payload = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        entries: dict[str, list[dict]] = {}
        per_prompt = _collect_per_prompt(payload, cfg.keys)
        if per_prompt:
            entries['per_prompt'] = _per_prompt_cis(per_prompt, cfg.n_boot, cfg.seed)
        if cfg.include_count_rates:
            count_rows = _collect_count_rates(payload)
            if count_rows:
                entries['count_rates'] = _count_rate_cis(count_rows, cfg.n_boot, cfg.seed)
        if entries:
            by_file[path.stem] = entries

    result = {
        'n_boot': cfg.n_boot,
        'seed': cfg.seed,
        'results_dir': str(cfg.results_dir),
        'results_glob': cfg.results_glob,
        'keys': list(cfg.keys),
        'n_files_scanned': len(paths),
        'n_files_with_entries': len(by_file),
        'by_file': by_file,
    }
    save_results(result, f'bootstrap_cis_{cfg.source}', 'all_models', results_dir=cfg.results_dir)
    return result


def _collect_per_prompt(payload: Any, keys: tuple[str, ...]) -> list[tuple[str, list[float]]]:
    """Walk the JSON payload, collecting (path, values) pairs whose key is in ``keys``."""
    out: list[tuple[str, list[float]]] = []
    _walk_per_prompt(payload, [], keys, out)
    return out


def _walk_per_prompt(
    obj: Any,
    path: list[str],
    keys: tuple[str, ...],
    out: list[tuple[str, list[float]]],
) -> None:
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in keys and _is_numeric_list(v):
                out.append(('.'.join([*path, str(k)]), [float(x) for x in v]))
            else:
                _walk_per_prompt(v, [*path, str(k)], keys, out)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            _walk_per_prompt(v, [*path, f'[{i}]'], keys, out)


def _is_numeric_list(v: Any) -> bool:
    return isinstance(v, list) and bool(v) and all(isinstance(x, int | float | bool) for x in v)


def _collect_count_rates(payload: Any) -> list[tuple[str, int, int]]:
    """Walk the JSON payload, collecting nested ``(agree|correct, total)`` count pairs."""
    out: list[tuple[str, int, int]] = []
    _walk_counts(payload, [], out)
    return out


def _walk_counts(obj: Any, path: list[str], out: list[tuple[str, int, int]]) -> None:
    if isinstance(obj, dict):
        n_pos, total = _try_extract_counts(obj)
        if total > 0 and 0 <= n_pos <= total:
            out.append(('.'.join(path) or '<root>', n_pos, total))
        for k, v in obj.items():
            _walk_counts(v, [*path, str(k)], out)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            _walk_counts(v, [*path, f'[{i}]'], out)


def _try_extract_counts(obj: dict[str, Any]) -> tuple[int, int]:
    """Pull (n_pos, total) out of a dict if both keys are present and parseable; return (-1, 0) otherwise."""
    if 'total' not in obj or ('agree' not in obj and 'correct' not in obj):
        return -1, 0
    raw_pos: Any = obj['agree'] if 'agree' in obj else obj['correct']
    raw_total: Any = obj['total']
    try:
        n_pos = int(raw_pos)
        total = int(raw_total)
    except (TypeError, ValueError):
        return -1, 0
    return n_pos, total


def _per_prompt_cis(rows: list[tuple[str, list[float]]], n_boot: int, seed: int) -> list[dict]:
    """Bootstrap the mean of each per-prompt indicator list (non-paired)."""
    out: list[dict] = []
    for path, values in rows:
        arr = np.asarray(values, dtype=float)
        mean = float(arr.mean())
        rng = np.random.RandomState(seed)
        n = len(arr)
        boots = np.empty(n_boot, dtype=float)
        for i in range(n_boot):
            idx = rng.choice(n, n, replace=True)
            boots[i] = arr[idx].mean()
        lo, hi = np.percentile(boots, [2.5, 97.5])
        out.append({'path': path, 'n': n, 'mean': mean, 'bootstrap_ci_95': [float(lo), float(hi)]})
    return out


def _count_rate_cis(rows: list[tuple[str, int, int]], n_boot: int, seed: int) -> list[dict]:
    """Wilson + bootstrap 95% CI for any (n_pos, total) count pair."""
    out: list[dict] = []
    for path, n_pos, total in rows:
        rate = n_pos / total
        wlo, whi = _wilson_ci(n_pos, total)
        blo, bhi = _bootstrap_count_ci(n_pos, total, n_boot, seed)
        out.append(
            {
                'path': path,
                'n_pos': n_pos,
                'total': total,
                'rate': rate,
                'wilson_ci_95': [wlo, whi],
                'bootstrap_ci_95': [blo, bhi],
            }
        )
    return out


def _wilson_ci(n_pos: int, total: int, z: float = _WILSON_Z) -> tuple[float, float]:
    if total == 0:
        return _DEGENERATE_LO, _DEGENERATE_HI
    p = n_pos / total
    denom = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denom
    half = z * np.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denom
    return float(max(0.0, centre - half)), float(min(1.0, centre + half))


def _bootstrap_count_ci(n_pos: int, total: int, n_boot: int, seed: int) -> tuple[float, float]:
    if total == 0:
        return _DEGENERATE_LO, _DEGENERATE_HI
    rng = np.random.RandomState(seed)
    data = np.concatenate([np.ones(n_pos, dtype=np.int8), np.zeros(total - n_pos, dtype=np.int8)])
    boots = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        boots[i] = data[rng.choice(total, total, replace=True)].mean()
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(lo), float(hi)


def add_cli_args(parser: argparse.ArgumentParser) -> None:
    """Register CLI flags for this analysis on ``parser``."""
    parser.add_argument('--results-dir', type=Path, default=None, help='Override the default results directory.')
    parser.add_argument(
        '--results-glob',
        default=_DEFAULT_GLOB,
        help='Glob pattern restricting which JSON files to scan (e.g. "*_gemma_2_2b_it.json").',
    )
    parser.add_argument(
        '--keys',
        default=','.join(_DEFAULT_KEYS),
        help='Comma-separated per-prompt indicator keys to bootstrap.',
    )
    parser.add_argument('--n-boot', type=int, default=_DEFAULT_N_BOOT)
    parser.add_argument('--seed', type=int, default=RANDOM_SEED)
    parser.add_argument(
        '--source',
        default='all',
        help='Tag appended to the output filename: bootstrap_cis_<source>.json.',
    )
    parser.add_argument(
        '--no-count-rates',
        dest='include_count_rates',
        action='store_false',
        help='Skip Wilson + bootstrap CIs on nested {agree|correct, total} count pairs.',
    )
    parser.set_defaults(include_count_rates=True)


def from_args(args: argparse.Namespace) -> BootstrapCisConfig:
    """Build the validated config from a parsed argparse namespace."""
    keys = tuple(s for s in args.keys.split(',') if s) if args.keys else _DEFAULT_KEYS
    return BootstrapCisConfig(
        results_dir=args.results_dir or RESULTS_DIR,
        results_glob=args.results_glob,
        keys=keys,
        n_boot=args.n_boot,
        seed=args.seed,
        source=args.source,
        include_count_rates=args.include_count_rates,
    )
