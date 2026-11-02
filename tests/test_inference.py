import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from inference.images import (axis_starts, blend_weight, merge_tiles, normalized_image,
                              overlay_image, plan_images, save_image)
from inference.records import open_run, atomic_json
from inference.engine import predict_image


class InferenceTests(unittest.TestCase):
    def test_tile_coverage_odd_and_small_sizes(self):
        for length in [1, 63, 512, 513, 1004, 3072]:
            seen = np.zeros(length, dtype=bool)
            for start in axis_starts(length, 512, 128):
                seen[start:min(start+512, length)] = True
            self.assertTrue(seen.all())

    def test_invalid_overlap(self):
        for overlap in [-1, 512]:
            with self.assertRaises(ValueError):
                axis_starts(1024, 512, overlap)

    def test_blending_constant_probabilities(self):
        probability = np.array([.2, .8], dtype=np.float32)[:, None, None]
        tiles = [(y, x, np.broadcast_to(probability, (2, min(64, 101-y), min(64, 113-x))))
                 for y in axis_starts(101, 64, 16) for x in axis_starts(113, 64, 16)]
        output = merge_tiles(tiles, 2, 101, 113)
        np.testing.assert_allclose(output[1], .8, atol=2e-7)
        np.testing.assert_allclose(output.sum(0), 1., atol=2e-7)

    def test_uncovered_pixels_fail(self):
        with self.assertRaises(ValueError):
            merge_tiles([], 2, 10, 10)

    def test_weights_positive(self):
        self.assertTrue((blend_weight(64, 64) > 0).all())

    def test_normalization(self):
        image = Image.new('RGB', (3, 2), (255, 0, 0))
        x = normalized_image(image)
        self.assertEqual(x.shape, (3, 2, 3))
        self.assertEqual(x.dtype, np.float32)
        self.assertAlmostEqual(float(x[0, 0, 0]), (1-.485)/.229, places=5)

    def test_overlay_only_foreground(self):
        image = Image.new('RGB', (2, 2), (100, 100, 100))
        labels = np.array([[0, 1], [0, 1]], dtype=np.uint8)
        out = np.array(overlay_image(image, labels, .4))
        np.testing.assert_array_equal(out[0, 0], [100, 100, 100])
        np.testing.assert_array_equal(out[0, 1], [60, 162, 60])

    def test_clean_names_nested_and_unicode(self):
        with TemporaryDirectory() as tmp:
            source = Path(tmp)/'输入'
            (source/'sub').mkdir(parents=True)
            (source/'sub/叶片.JPG').touch()
            plan = plan_images(source, Path(tmp)/'结果')
            self.assertEqual(plan[0][2].name, '叶片_overlay.jpg')
            self.assertEqual(plan[0][2].parent.name, 'sub')

    def test_same_stem_collision(self):
        with TemporaryDirectory() as tmp:
            source = Path(tmp)/'input'
            source.mkdir()
            (source/'leaf.jpg').touch()
            (source/'leaf.png').touch()
            with self.assertRaises(ValueError):
                plan_images(source, Path(tmp)/'output')

    def test_output_inside_input_rejected(self):
        with TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                plan_images(tmp, Path(tmp)/'output')

    def test_unicode_jpeg_save_and_dimensions(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp)/'中文/叶片_overlay.jpg'
            save_image(Image.new('RGB', (37, 29)), path)
            with Image.open(path) as image:
                self.assertEqual(image.format, 'JPEG')
                self.assertEqual(image.size, (37, 29))
            self.assertFalse(list(path.parent.glob('*.tmp')))

    def test_manifest_resume_and_mismatch(self):
        with TemporaryDirectory() as tmp:
            output = Path(tmp)/'out'
            state, manifest = open_run(output, {'mode': 'full'}, False)
            state['completed']['a.jpg'] = {'test': 1}
            atomic_json(state, manifest)
            self.assertIn('a.jpg', open_run(output, {'mode':'full'}, True)[0]['completed'])
            with self.assertRaises(ValueError):
                open_run(output, {'mode':'tile'}, True)
            with self.assertRaises(FileExistsError):
                open_run(output, {'mode':'full'}, False)

    def test_full_mode_calls_once_no_resize(self):
        image = Image.new('RGB', (113, 101))
        with patch('inference.engine.forward_image', return_value=np.zeros((101,113), dtype=np.uint8)) as forward:
            labels, probability = predict_image(None, image, None, True, 'full')
        self.assertEqual(forward.call_count, 1)
        self.assertIs(forward.call_args.args[1], image)
        self.assertEqual(labels.shape, (101,113))
        self.assertIsNone(probability)

    def test_tile_probabilities_and_threshold(self):
        image = Image.new('RGB', (113, 101))
        def forward(model, crop, device, amp, probabilities):
            self.assertTrue(probabilities)
            return np.broadcast_to(np.array([.2,.8], dtype=np.float32)[:,None,None], (2,crop.height,crop.width)).copy()
        with patch('inference.engine.forward_image', side_effect=forward):
            labels, prob = predict_image(None, image, None, True, 'tile', 64, 16, 2, .7, True)
        self.assertTrue((labels == 1).all())
        self.assertEqual(prob.shape, (2,101,113))


if __name__ == '__main__':
    unittest.main()
