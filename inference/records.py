import hashlib
import json
import os
from pathlib import Path
import uuid


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(payload, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    try:
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def open_run(output, config, resume, overwrite=False):
    output = Path(output)
    manifest = output / 'meta' / 'run.json'
    if resume and overwrite:
        raise ValueError('Choose either resume or overwrite')
    if overwrite:
        state = dict(config=config, completed={})
        atomic_json(state, manifest)
        (output / 'meta' / 'per_image.csv').unlink(missing_ok=True)
        return state, manifest
    if manifest.exists():
        if not resume:
            raise FileExistsError('Existing run: use --resume or a new --output')
        state = json.loads(manifest.read_text(encoding='utf-8'))
        if state['config'] != config:
            raise ValueError('Resume requires identical weights, source and inference settings')
    else:
        if output.exists() and any(output.iterdir()):
            raise FileExistsError('Output is nonempty and has no run manifest; choose a new directory')
        state = dict(config=config, completed={})
        atomic_json(state, manifest)
    return state, manifest
