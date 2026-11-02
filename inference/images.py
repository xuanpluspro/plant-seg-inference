from pathlib import Path
import os
import uuid

import numpy as np
from PIL import Image

EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
MEAN = np.array([.485, .456, .406], dtype=np.float32)[:, None, None]
STD = np.array([.229, .224, .225], dtype=np.float32)[:, None, None]


def normalized_image(image):
    array = np.array(image.convert('RGB'), dtype=np.float32).transpose(2, 0, 1) / 255.
    return np.ascontiguousarray((array - MEAN) / STD)


def axis_starts(length, tile, overlap):
    if length < 1 or tile < 64 or not 0 <= overlap < tile:
        raise ValueError('Require positive image size, tile >=64, 0 <= overlap < tile')
    if length <= tile:
        return [0]
    starts = list(range(0, length - tile + 1, tile - overlap))
    if starts[-1] != length - tile:
        starts.append(length - tile)
    return starts


def blend_weight(height, width):
    # Positive triangular weights: even image-edge pixels retain nonzero support.
    y = 1 - np.abs((np.arange(height, dtype=np.float32) + .5) * 2 / height - 1)
    x = 1 - np.abs((np.arange(width, dtype=np.float32) + .5) * 2 / width - 1)
    return np.maximum(y[:, None] * x[None, :], 1e-3)


def merge_tiles(tiles, classes, height, width):
    total = np.zeros((classes, height, width), dtype=np.float32)
    weight = np.zeros((height, width), dtype=np.float32)
    for y, x, probability in tiles:
        h, w = probability.shape[-2:]
        if probability.shape[0] != classes or y < 0 or x < 0 or y+h > height or x+w > width:
            raise ValueError('Invalid tile bounds or channel count')
        window = blend_weight(h, w)
        total[:, y:y+h, x:x+w] += probability * window
        weight[y:y+h, x:x+w] += window
    if np.any(weight == 0):
        raise ValueError('Tile layout left uncovered pixels')
    total /= weight[None]
    return total


def plan_images(source, output):
    source, output = Path(source).resolve(), Path(output).resolve()
    if not source.exists():
        raise FileNotFoundError(source)
    if source.is_dir() and (source == output or source in output.parents):
        raise ValueError('Output must be outside the input directory')
    paths = sorted(p for p in source.rglob('*') if p.is_file() and p.suffix.lower() in EXTENSIONS) if source.is_dir() else [source]
    if not paths or any(p.suffix.lower() not in EXTENSIONS for p in paths):
        raise ValueError('No supported input images')
    result, seen = [], set()
    for path in paths:
        relative = path.relative_to(source) if source.is_dir() else Path(path.name)
        # Remove the original extension: leaf.jpg -> leaf_overlay.jpg.
        target = output / relative.parent / (relative.stem + '_overlay.jpg')
        key = str(target).casefold()
        if key in seen:
            raise ValueError(f'Output name collision; rename same-stem inputs: {path}')
        seen.add(key)
        result.append((path, relative, target))
    return result


def overlay_image(image, labels, alpha=.4):
    pixels = np.array(image.convert('RGB'))
    if labels.shape != pixels.shape[:2] or not 0 <= alpha <= 1:
        raise ValueError('Mask shape or overlay alpha invalid')
    palette = np.array([[(37*i) % 256, (97*i+158) % 256, (173*i+83) % 256]
                        for i in range(256)], dtype=np.uint8)
    palette[1] = [0, 255, 0]
    active = labels > 0
    pixels[active] = np.round(pixels[active].astype(np.float32) * (1-alpha)
                              + palette[labels[active]].astype(np.float32) * alpha).astype(np.uint8)
    return Image.fromarray(pixels)


def save_image(image, destination, quality=90):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + '.' + uuid.uuid4().hex + '.tmp')
    try:
        with temporary.open('wb') as stream:
            if destination.suffix.lower() in ('.jpg', '.jpeg'):
                image.save(stream, format='JPEG', quality=quality, optimize=True)
            else:
                image.save(stream, format='PNG')
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
