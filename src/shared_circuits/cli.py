"""Command-line interface for shared-circuits analyses."""

import argparse
import importlib
from types import ModuleType
from typing import Final

ANALYSES: Final[tuple[str, ...]] = (
    'activation-patching',
    'attribution-patching',
    'bootstrap-cis',
    'breadth',
    'causal-ablation',
    'circuit-overlap',
    'direction-analysis',
    'dla-instructed-lying',
    'dpo-antisyc',
    'faithfulness',
    'head-zeroing',
    'layer-strat-null',
    'linear-probe-sae-alignment',
    'logit-lens',
    'mlp-ablation',
    'mlp-mediation',
    'norm-matched',
    'nq-replication',
    'opinion-causal',
    'opinion-circuit-transfer',
    'path-patching',
    'probe-transfer',
    'projection-ablation',
    'reverse-projection',
    'sae-feature-overlap',
    'sae-k-sensitivity',
    'sae-sentiment-control',
    'steering',
    'triple-intersection',
)


def _module_for(slug: str) -> ModuleType:
    """Resolve a kebab-case analysis slug to its module under ``shared_circuits.analyses``."""
    return importlib.import_module(f'shared_circuits.analyses.{slug.replace("-", "_")}')


def main(argv: list[str] | None = None) -> None:
    """CLI entrypoint registered as ``shared-circuits``."""
    parser = argparse.ArgumentParser(prog='shared-circuits')
    sub = parser.add_subparsers(dest='command', required=True)

    run_parser = sub.add_parser('run', help='Run a registered analysis')
    run_sub = run_parser.add_subparsers(dest='analysis', required=True)
    for slug in ANALYSES:
        mod = _module_for(slug)
        # walk past a leading blank line so multiline `"""\nSummary...` docstrings render
        help_text = next(
            (line.strip() for line in (mod.__doc__ or slug).splitlines() if line.strip()),
            slug,
        )
        analysis_parser = run_sub.add_parser(slug, help=help_text, description=help_text)
        mod.add_cli_args(analysis_parser)

    args = parser.parse_args(argv)
    mod = _module_for(args.analysis)
    cfg = mod.from_args(args)
    mod.run(cfg)


if __name__ == '__main__':
    main()
