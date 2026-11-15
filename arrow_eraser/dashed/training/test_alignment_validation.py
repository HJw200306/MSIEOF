"""Regression checks for geometry and post-processing, independent of weights."""
import unittest

import numpy as np
import torch

from arrow_eraser.arrow_earser_val import (
    PreprocessorWrapper, best_translation_diagnostic, get_full_image_aug,
    legacy_refine_prediction, mask_metrics, refine_prediction, run_inference,
)


class DarkPixelModel(torch.nn.Module):
    """Predict a known black pixel pattern after ImageNet normalization."""
    def forward(self, x):
        return torch.stack((x[:, 0], -x[:, 0]), dim=1)


class AlignmentTests(unittest.TestCase):
    def test_full_forward_preserves_original_coordinates(self):
        for h, w in ((1, 1), (31, 65), (127, 258), (512, 512), (513, 517)):
            with self.subTest(shape=(h, w)):
                gt = np.zeros((h, w), np.uint8)
                gt[0, 0] = gt[-1, -1] = gt[h // 2, w // 3] = 255
                rgb = np.full((h, w, 3), 255, np.uint8)
                rgb[gt > 0] = 0
                padded = get_full_image_aug((255, 255, 255))(image=rgb, mask=gt)["image"]
                np.testing.assert_array_equal(padded[:h, :w], rgb)
                self.assertTrue(np.all(padded[h:, :] == 255))
                self.assertTrue(np.all(padded[:, w:] == 255))
                pred, padded_gt = run_inference(DarkPixelModel(), rgb, gt,
                    get_full_image_aug(), PreprocessorWrapper(), "cpu")
                self.assertEqual(pred.shape[0] % 32, 0)
                self.assertEqual(pred.shape[1] % 32, 0)
                np.testing.assert_array_equal(pred[:h, :w], gt > 0)
                np.testing.assert_array_equal(pred, padded_gt)

    def test_refinement_keeps_pixels_and_has_no_directional_bias(self):
        raw = np.zeros((41, 51), np.uint8)
        raw[10, 5:30] = 1
        raw[20:29, 36] = 1
        raw[30, 9] = 1
        raw[0, 0] = 1
        refined = refine_prediction(raw)
        self.assertTrue(np.all(refined[raw > 0] == 1))
        for axis in (0, 1):
            np.testing.assert_array_equal(
                refine_prediction(np.flip(raw, axis)), np.flip(refined, axis))
        np.testing.assert_array_equal(refine_prediction(raw.T), refined.T)

    def test_legacy_thin_line_regression(self):
        raw = np.zeros((41, 51), np.uint8)
        raw[20, 10:41] = 1
        center_y = np.argwhere(refine_prediction(raw))[:, 0].mean()
        legacy_y = np.argwhere(legacy_refine_prediction(raw))[:, 0].mean()
        self.assertAlmostEqual(center_y, 20)
        self.assertGreater(legacy_y, center_y + .5)

    def test_diagnostic_shift_is_correct_and_does_not_mutate(self):
        gt = np.zeros((30, 30), np.uint8)
        gt[10, 10:15] = 1
        gt[11:15, 10] = 1
        pred = np.zeros_like(gt)
        pred[9, 12:17] = 1
        pred[10:14, 12] = 1
        before = pred.copy()
        result = best_translation_diagnostic(pred, gt, radius=4)
        self.assertEqual((result["dx"], result["dy"]), (-2, 1))
        self.assertAlmostEqual(result["best_iou"], 1.)
        np.testing.assert_array_equal(pred, before)

    def test_diagnostic_does_not_reward_discarding_edge_pixels(self):
        gt = np.zeros((10, 10), np.uint8)
        gt[5, 5] = 1
        pred = np.zeros_like(gt)
        pred[5, 6] = pred[5, 0] = 1
        result = best_translation_diagnostic(pred, gt, radius=2)
        self.assertAlmostEqual(result["best_iou"], .5)

    def test_empty_masks(self):
        empty = np.zeros((10, 10), np.uint8)
        self.assertEqual(mask_metrics(empty, empty)["iou"], 1.)
        self.assertIsNone(best_translation_diagnostic(empty, empty)["dx"])


if __name__ == "__main__":
    unittest.main()
