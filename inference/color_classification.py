"""Optional foreground-only color classification; no neural-network execution."""
import csv
import os
import pickle
from pathlib import Path
import uuid

import numpy as np
from PIL import Image

COLOR_FIELDS = ['foreground_pixels', 'green_pixels', 'senescent_pixels',
                'vegetation_fraction', 'green_fraction', 'senescent_fraction',
                'senescent_ratio', 'senescent_percent', 'color_seconds']


class ColorClassifier:
    def __init__(self, path, trusted=False, green_label=1, senescent_label=0, threshold=.5):
        if not trusted:
            raise ValueError('Color pickle requires --trust-color-model for your own trusted model')
        if green_label == senescent_label or not 0 < threshold < 1:
            raise ValueError('Distinct class labels and threshold in (0,1) required')
        # Imported only when color classification is requested.
        from .color_features import get_vegetation_features
        self.features = get_vegetation_features
        with Path(path).open('rb') as stream:
            self.model = pickle.load(stream)
        if not hasattr(self.model, 'predict_proba'):
            raise ValueError('Classifier requires predict_proba; SVM must be probability=True')
        classes = np.asarray(getattr(self.model, 'classes_', []))
        if classes.shape != (2,) or set(classes.tolist()) != {green_label, senescent_label}:
            raise ValueError(f'Classifier classes {classes} do not match supplied labels')
        if getattr(self.model, 'n_features_in_', None) != 14:
            raise ValueError('Requires 14-feature model; 21D and unknown feature schemas are unsupported')
        names = getattr(self.model, 'feature_names_in_', None)
        if names is not None:
            raise ValueError('Named-feature estimator unsupported; export an array-trained 14D model')
        self.green_column = int(np.flatnonzero(classes == green_label)[0])
        self.threshold = threshold

    def predict_green(self, pixels):
        probability = np.asarray(self.model.predict_proba(self.features(pixels)))
        if probability.shape != (len(pixels), 2) or not np.isfinite(probability).all():
            raise ValueError('Invalid classifier probability output')
        if np.any((probability < 0) | (probability > 1)) or not np.allclose(probability.sum(1), 1, atol=1e-4):
            raise ValueError('Classifier output is not normalized probability')
        return probability[:, self.green_column] >= self.threshold


def classify_foreground(rgb, foreground, classifier, chunk_size=65536):
    rgb, foreground = np.asarray(rgb), np.asarray(foreground)
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3 or foreground.shape != rgb.shape[:2]:
        raise ValueError('Expected uint8 RGB and same-size foreground mask')
    if chunk_size < 1:
        raise ValueError('color chunk size must be positive')
    foreground = foreground.astype(bool)
    # 0 background, 1 green, 2 senescent; only original foreground is classified.
    labels = np.zeros(foreground.shape, dtype=np.uint8)
    flat_rgb, flat_mask, flat_output = rgb.reshape(-1, 3), foreground.ravel(), labels.ravel()
    for start in range(0, len(flat_mask), chunk_size):
        indices = np.flatnonzero(flat_mask[start:start+chunk_size]) + start
        if len(indices):
            green = np.asarray(classifier.predict_green(flat_rgb[indices]), dtype=bool)
            if green.shape != (len(indices),):
                raise ValueError('Invalid classifier prediction length')
            flat_output[indices] = np.where(green, 1, 2)
    n_green, n_senescent = int((labels == 1).sum()), int((labels == 2).sum())
    n_foreground, total = n_green+n_senescent, labels.size
    ratio = n_senescent/n_foreground if n_foreground else None
    stats = dict(foreground_pixels=n_foreground, green_pixels=n_green, senescent_pixels=n_senescent,
                 vegetation_fraction=n_foreground/total, green_fraction=n_green/total,
                 senescent_fraction=n_senescent/total, senescent_ratio=ratio,
                 senescent_percent=ratio*100 if ratio is not None else None)
    return labels, stats


def color_mask(labels):
    image = Image.fromarray(labels, mode='P')
    palette = [0,0,0, 0,180,70, 255,255,0] + [0]*759
    image.putpalette(palette)
    return image


def color_overlay(image, labels, alpha):
    pixels = np.array(image.convert('RGB'))
    palette = np.array([[0,0,0],[0,180,70],[255,255,0]], dtype=np.uint8)
    active = labels > 0
    pixels[active] = np.round((1-alpha)*pixels[active].astype(np.float32)+alpha*palette[labels[active]]).astype(np.uint8)
    return Image.fromarray(pixels)


def export_color_csv(output, completed):
    path = Path(output)/'meta/color_metrics.csv'
    temporary = path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    try:
        with temporary.open('w', newline='', encoding='utf-8-sig') as stream:
            writer = csv.DictWriter(stream, fieldnames=['image']+COLOR_FIELDS)
            writer.writeheader()
            for key, row in completed.items():
                if 'color_stats' in row:
                    writer.writerow(dict(image=key, **row['color_stats']))
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
