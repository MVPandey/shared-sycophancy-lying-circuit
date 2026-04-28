"""Manual pipeline-parallelism layout for HookedTransformer across multiple GPUs."""


def distribute_pipeline_parallel(
    model: 'HookedTransformer',
    device: str,
    n_devices: int,
) -> 'HookedTransformer':
    """
    Split a HookedTransformer's blocks across `n_devices` of `device` type.

    TransformerLens's `.to()` does not distribute — each module must be moved to
    its target device manually. (TL's `devices.get_device_for_block_index` was
    removed in newer versions, so we reimplement the layer→device mapping here.)

    Args:
        model: A HookedTransformer already on a single device or CPU.
        device: Device type string (e.g. ``'cuda'``); the function appends ``:i``.
        n_devices: Number of devices to split blocks across.

    Returns:
        The same model with submodules moved to their target devices in-place.

    """
    n_layers = model.cfg.n_layers
    layers_per_device = (n_layers + n_devices - 1) // n_devices  # ceil

    def dev_for_block(block_idx: int) -> str:
        d = min(block_idx // layers_per_device, n_devices - 1)
        return f'{device}:{d}'

    for i, block in enumerate(model.blocks):
        block.to(dev_for_block(i))
    first_dev = dev_for_block(0)
    last_dev = dev_for_block(n_layers - 1)
    model.embed.to(first_dev)
    if hasattr(model, 'pos_embed') and model.pos_embed is not None:
        model.pos_embed.to(first_dev)
    if hasattr(model, 'hook_embed'):
        model.hook_embed.to(first_dev)
    if hasattr(model, 'ln_final') and model.ln_final is not None:
        model.ln_final.to(last_dev)
    model.unembed.to(last_dev)
    return model
