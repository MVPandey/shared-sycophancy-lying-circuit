"""JSON persistence for experiment results."""

import json
from pathlib import Path

# parents[3] walks src/shared_circuits/experiment/results.py up to the repo root.
RESULTS_DIR: Path = Path(__file__).resolve().parents[3] / 'experiments' / 'results'


def model_slug(model_name: str) -> str:
    """
    Convert a HF model name into a filesystem-safe slug for filenames.

    Args:
        model_name: HuggingFace model name (may contain ``/`` and ``-``).

    Returns:
        Slugified string suitable for filenames.

    """
    return model_name.replace('/', '_').replace('-', '_')


def save_results(
    data: dict,
    experiment_name: str,
    model_name: str,
    results_dir: Path | None = None,
) -> Path:
    """
    Save experiment results as JSON.

    Args:
        data: Results dictionary to serialize.
        experiment_name: Name of the experiment (e.g. ``'circuit_overlap'``).
        model_name: Model identifier for the filename slug.
        results_dir: Override the default results directory.

    Returns:
        Path to the saved file.

    """
    out_dir = results_dir or RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    slug = model_slug(model_name)
    out_path = out_dir / f'{experiment_name}_{slug}.json'
    with open(out_path, 'w') as f:
        json.dump(data, f, indent=2)
    return out_path


def load_results(
    experiment_name: str,
    model_name: str,
    results_dir: Path | None = None,
) -> dict:
    """
    Load experiment results from JSON.

    Args:
        experiment_name: Name of the experiment.
        model_name: Model identifier for the filename slug.
        results_dir: Override the default results directory.

    Returns:
        Loaded results dictionary.

    Raises:
        FileNotFoundError: If the results file doesn't exist.

    """
    out_dir = results_dir or RESULTS_DIR
    slug = model_slug(model_name)
    out_path = out_dir / f'{experiment_name}_{slug}.json'
    with open(out_path) as f:
        return json.load(f)
