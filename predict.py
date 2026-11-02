"""Batch segmentation: CUDA full images or explicit probability-blended tiles."""
import argparse
import csv
import json
import logging
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image

from inference.images import plan_images, overlay_image, save_image
from inference.records import sha256_file, open_run, atomic_json
from inference.postprocess import keep_largest_component


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--weights', type=Path, required=True)
    p.add_argument('--input', '--data-path', dest='source', type=Path, required=True)
    p.add_argument('--output', type=Path, help='Default: sibling <input stem>visual')
    p.add_argument('--mode', choices=['full', 'tile'], default='full')
    p.add_argument('--device', default='cuda:0', help='CUDA device only; no automatic CPU fallback')
    p.add_argument('--amp', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--tile-size', type=int, default=512)
    p.add_argument('--overlap', type=int, default=128)
    p.add_argument('--alpha', type=float, default=.4)
    p.add_argument('--jpg-quality', type=int, default=90)
    p.add_argument('--threshold', type=float, help='Optional binary foreground probability threshold; default argmax')
    p.add_argument('--save-mask', action='store_true', help='Optional PNG mask; binary IDs encoded 0/255')
    p.add_argument('--save-probabilities', action='store_true', help='Optional float32 CxHxW NPY; may be large')
    p.add_argument('--largest-component', action=argparse.BooleanOptionalAction, default=True,
                   help='Keep only the largest non-background connected region (default: enabled)')
    p.add_argument('--connectivity', type=int, choices=[4, 8], default=8)
    writes = p.add_mutually_exclusive_group()
    writes.add_argument('--resume', action='store_true', help='Skip completed, unchanged inputs after checking run settings')
    writes.add_argument('--overwrite', action='store_true', help='Reprocess all inputs and replace matching outputs in the existing output directory')
    return p


def main(args):
    if not 0 <= args.alpha <= 1 or not 1 <= args.jpg_quality <= 100:
        raise ValueError('Require alpha in [0,1] and JPG quality in [1,100]')
    if args.tile_size < 64 or args.tile_size % 32 or not 0 <= args.overlap < args.tile_size:
        raise ValueError('Require tile-size >=64 divisible by 32, and 0 <= overlap < tile-size')
    if args.threshold is not None and not 0 < args.threshold < 1:
        raise ValueError('Threshold must be in (0,1)')
    if args.largest_component:
        from scipy import ndimage  # Fail early if the postprocessing dependency is missing.
    import torch
    from inference.checkpoints import load_model
    from inference.engine import predict_image
    device = torch.device(args.device)
    if device.type != 'cuda' or not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU required. This tool does not fall back to CPU.')
    torch.cuda.set_device(device)
    args.source = args.source.resolve()
    output = (args.output or args.source.with_name(args.source.stem+'visual')).resolve()
    plan = plan_images(args.source, output)
    config = dict(weights_sha256=sha256_file(args.weights), source=str(args.source),
                  mode=args.mode, amp=args.amp, tile_size=args.tile_size, overlap=args.overlap,
                  alpha=args.alpha, jpg_quality=args.jpg_quality, threshold=args.threshold,
                  save_mask=args.save_mask, save_probabilities=args.save_probabilities,
                  largest_component=args.largest_component, connectivity=args.connectivity,
                  upstream=json.loads((Path(__file__).parent/'UPSTREAM.json').read_text())['commit'])
    state, manifest = open_run(output, config, args.resume, overwrite=args.overwrite)
    logger = logging.getLogger('inference')
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    for handler in (logging.StreamHandler(sys.stdout), logging.FileHandler(output/'meta/console.log', mode='w' if args.overwrite else 'a', encoding='utf-8')):
        handler.setFormatter(logging.Formatter('%(asctime)s | %(message)s'))
        logger.addHandler(handler)
    try:
        model, model_config = load_model(args.weights, device)
        classes = model_config['num_classes']
        if args.threshold is not None and classes != 2:
            raise ValueError('--threshold is supported only for binary checkpoints')
        logger.info('GPU=%s | images=%d | mode=%s | AMP=%s | output=%s',
                    torch.cuda.get_device_name(device), len(plan), args.mode, args.amp, output)
        for index, (path, relative, target) in enumerate(plan, 1):
            key = relative.as_posix()
            stat = path.stat()
            signature = dict(size=stat.st_size, mtime_ns=stat.st_mtime_ns)
            previous = state['completed'].get(key)
            extra = output/'extras'/relative.parent
            mask_path = extra/(relative.stem+'_mask.png')
            prob_path = extra/(relative.stem+'_prob.npy')
            required = [target] + ([mask_path] if args.save_mask else []) + ([prob_path] if args.save_probabilities else [])
            if previous and previous['input'] == signature and all(p.is_file() for p in required):
                logger.info('[%d/%d] skip completed %s', index, len(plan), key)
                continue
            if not previous and not args.resume and not args.overwrite and any(p.exists() for p in required):
                raise FileExistsError(f'Untracked output exists: {target}; use a fresh output directory')
            with Image.open(path) as source:
                image = source.convert('RGB')
            logger.info('[%d/%d] %s | %dx%d', index, len(plan), key, image.width, image.height)
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)
            start = time.perf_counter()
            try:
                labels, probabilities = predict_image(model, image, device, args.amp, args.mode,
                    args.tile_size, args.overlap, classes, args.threshold, args.save_probabilities)
            except torch.OutOfMemoryError:
                logger.error('CUDA OOM for %s. No resizing, tiling or CPU fallback was performed. '
                             'For tiles, rerun with --mode tile and a NEW --output.', key)
                raise
            torch.cuda.synchronize(device)
            seconds = time.perf_counter()-start
            peak = torch.cuda.max_memory_allocated(device)/1024**3
            post_start = time.perf_counter()
            if args.largest_component:
                labels, component_stats = keep_largest_component(labels, args.connectivity)
                logger.info('components=%d | foreground=%d -> %d pixels | removed=%d',
                            component_stats['components'], component_stats['foreground_before'],
                            component_stats['foreground_after'], component_stats['removed_pixels'])
            else:
                component_stats = {}
            postprocess_seconds = time.perf_counter()-post_start
            save_start = time.perf_counter()
            if args.save_mask:
                save_image(Image.fromarray(labels*255 if classes == 2 else labels), mask_path)
            if args.save_probabilities:
                extra.mkdir(parents=True, exist_ok=True)
                temporary = prob_path.with_suffix('.npy.tmp')
                with temporary.open('wb') as stream:
                    np.save(stream, probabilities, allow_pickle=False)
                temporary.replace(prob_path)
            # Overlay saved last; record is written only after all requested files succeed.
            save_image(overlay_image(image, labels, args.alpha), target, args.jpg_quality)
            state['completed'][key] = dict(input=signature, width=image.width, height=image.height,
                inference_seconds=seconds, postprocess_seconds=postprocess_seconds,
                component_stats=component_stats, save_seconds=time.perf_counter()-save_start,
                peak_allocated_gib=peak, overlay=str(target.relative_to(output)))
            atomic_json(state, manifest)
            logger.info('inference=%.2fs | peak=%.2f GiB | saved=%s', seconds, peak, target.name)
        with (output/'meta/per_image.csv').open('w', newline='', encoding='utf-8-sig') as stream:
            fields = ['image','width','height','inference_seconds','postprocess_seconds','save_seconds','peak_allocated_gib','overlay']
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for key, row in state['completed'].items():
                writer.writerow(dict(image=key, **{k: row[k] for k in fields if k != 'image'}))
        logger.info('Finished. Results: %s', output)
    except Exception:
        logger.exception('Inference stopped; completed outputs are retained. Use --resume with the same settings.')
        raise


if __name__ == '__main__':
    main(parser().parse_args())
