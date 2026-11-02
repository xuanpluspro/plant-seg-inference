"""Checkpoint I/O. No model imports or GPU initialization for slimming."""
import os
from pathlib import Path
import uuid

FORMAT = 'plant_seg_lab_mpm5531_sparse_drm_cbm_v4'
TRAINING_KEYS = {'optimizer', 'scheduler', 'scaler', 'rng', 'history'}


def read_checkpoint(path):
    import torch
    with Path(path).open('rb') as stream:
        checkpoint = torch.load(stream, map_location='cpu', weights_only=True)
    if not isinstance(checkpoint, dict) or checkpoint.get('format') != FORMAT:
        raise ValueError('Expected plant-seg-lab v4 checkpoint; older architectures need matching source code')
    if 'model' not in checkpoint or 'model_config' not in checkpoint:
        raise ValueError('Checkpoint requires model and model_config; plain state_dict is unsupported')
    return checkpoint


def atomic_save(payload, path):
    import torch
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    try:
        with temporary.open('wb') as stream:
            torch.save(payload, stream)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def inference_checkpoint(checkpoint):
    import torch
    result = {k: v for k, v in checkpoint.items() if k not in TRAINING_KEYS}
    result['checkpoint_kind'] = 'inference'
    result['model'] = {k: v.detach().cpu().clone() if isinstance(v, torch.Tensor) else v
                       for k, v in checkpoint['model'].items()}
    return result


def load_model(path, device):
    from model import OSSSeg
    checkpoint = read_checkpoint(path)
    model = OSSSeg(**checkpoint['model_config'])
    model.load_state_dict(checkpoint['model'], strict=True)
    return model.to(device).eval(), checkpoint['model_config']
