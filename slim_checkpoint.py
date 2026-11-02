"""Strip resume-only state on CPU; never train or change tensor precision."""
import argparse
from pathlib import Path

from inference.checkpoints import read_checkpoint, inference_checkpoint, atomic_save


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--weights', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.weights.resolve() == args.output.resolve() or args.output.exists():
        raise FileExistsError('Choose a new output file; source is preserved')
    import torch
    original = read_checkpoint(args.weights)
    compact = inference_checkpoint(original)
    # Validate before publishing the final file.
    temporary = args.output.with_name(args.output.name+'.verify.tmp')
    if temporary.exists():
        raise FileExistsError(f'Remove stale verification file or choose a new output: {temporary}')
    try:
        atomic_save(compact, temporary)
        saved = read_checkpoint(temporary)
        if saved['model'].keys() != original['model'].keys():
            raise ValueError('Model key mismatch')
        for name, tensor in original['model'].items():
            if isinstance(tensor, torch.Tensor):
                actual = saved['model'][name]
                if actual.dtype != tensor.dtype or actual.shape != tensor.shape:
                    raise ValueError(f'Tensor metadata changed: {name}')
                torch.testing.assert_close(actual, tensor, rtol=0, atol=0, equal_nan=True)
        temporary.replace(args.output)
    finally:
        temporary.unlink(missing_ok=True)
    print(f'{args.weights.stat().st_size/2**20:.2f} -> {args.output.stat().st_size/2**20:.2f} MiB')
    print(f'Weights verified, precision unchanged; source retained. Saved: {args.output}')


if __name__ == '__main__':
    main()
