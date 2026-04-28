"""Command-line interface for shared-circuits analyses."""

from __future__ import annotations

import argparse
import importlib
from types import ModuleType
from typing import Final

ANALYSES: Final[tuple[str, ...]] = (
    'circuit-overlap',
    'causal-ablation',
    'attribution-patching',
    'breadth',
    'path-patching',
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
        help_text = (mod.__doc__ or slug).splitlines()[0]
        analysis_parser = run_sub.add_parser(slug, help=help_text, description=help_text)
        mod.add_cli_args(analysis_parser)

    args = parser.parse_args(argv)
    mod = _module_for(args.analysis)
    cfg = mod.from_args(args)
    mod.run(cfg)


if __name__ == '__main__':
    main()
