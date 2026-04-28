"""
Reference analyses dispatched by the ``shared-circuits`` CLI.

Each module in this package exposes the same surface — a frozen Pydantic config,
a ``run(cfg)`` entrypoint, and an ``add_cli_args``/``from_args`` pair used by
:mod:`shared_circuits.cli` to register the analysis as a subcommand.
"""

__all__ = [
    'attribution_patching',
    'breadth',
    'causal_ablation',
    'circuit_overlap',
    'path_patching',
]
