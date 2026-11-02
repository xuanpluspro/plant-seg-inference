import unittest
import numpy as np

from inference.postprocess import keep_largest_component


class ComponentTests(unittest.TestCase):
    def test_small_fragments_removed(self):
        labels = np.zeros((12, 12), dtype=np.uint8)
        labels[2:6, 2:6] = 1
        labels[9, 9] = 1
        result, stats = keep_largest_component(labels)
        self.assertEqual(int(result.sum()), 16)
        self.assertEqual(stats, dict(components=2, foreground_before=17,
                                    foreground_after=16, removed_pixels=1))
        self.assertEqual(int(labels.sum()), 17)  # Input is untouched.

    def test_empty(self):
        result, stats = keep_largest_component(np.zeros((4, 4), dtype=np.uint8))
        self.assertFalse(result.any())
        self.assertEqual(stats["components"], 0)

    def test_all_foreground(self):
        labels = np.ones((4, 7), dtype=np.uint8)
        result, stats = keep_largest_component(labels)
        np.testing.assert_array_equal(result, labels)
        self.assertEqual(stats["removed_pixels"], 0)

    def test_diagonal_connectivity(self):
        labels = np.eye(4, dtype=np.uint8)
        result8, stats8 = keep_largest_component(labels, 8)
        result4, stats4 = keep_largest_component(labels, 4)
        self.assertEqual(stats8["components"], 1)
        self.assertEqual(int(result8.sum()), 4)
        self.assertEqual(stats4["components"], 4)
        self.assertEqual(int(result4.sum()), 1)
        self.assertEqual(result4[0, 0], 1)

    def test_multiclass_union_preserves_class_ids(self):
        labels = np.array([[1, 2, 0, 3], [1, 2, 0, 0]], dtype=np.uint8)
        result, stats = keep_largest_component(labels, 4)
        np.testing.assert_array_equal(result, [[1, 2, 0, 0], [1, 2, 0, 0]])
        self.assertEqual(stats["foreground_after"], 4)

    def test_holes_not_filled(self):
        labels = np.ones((5, 5), dtype=np.uint8)
        labels[2, 2] = 0
        result, _ = keep_largest_component(labels)
        self.assertEqual(result[2, 2], 0)

    def test_invalid_inputs(self):
        for labels, connectivity in [(np.ones((2,2,2), dtype=np.uint8),8),
                                     (np.ones((2,2), dtype=float),8),
                                     (np.ones((2,2), dtype=np.uint8),6)]:
            with self.assertRaises(ValueError):
                keep_largest_component(labels, connectivity)

    def test_cli_defaults_and_opt_out(self):
        from predict import parser
        args = parser().parse_args(["--weights","model.pth","--input","images"])
        self.assertTrue(args.largest_component)
        self.assertEqual(args.connectivity, 8)
        args = parser().parse_args(["--weights","model.pth","--input","images",
                                    "--no-largest-component","--connectivity","4"])
        self.assertFalse(args.largest_component)
        self.assertEqual(args.connectivity, 4)


if __name__ == "__main__":
    unittest.main()
