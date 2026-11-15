import unittest
import numpy as np
from comparison_metrics import counts, summarize, fuse_and_erase


class MetricsTest(unittest.TestCase):
    def test_both_models_protect_bonds(self):
        rgb = np.zeros((1,4,3),np.uint8)
        solid = np.array([[1,2,0,1]],np.uint8)
        dashed = np.array([[2,1,1,0]],np.uint8)
        pred, erased, out = fuse_and_erase(rgb, [solid,dashed])
        np.testing.assert_array_equal(pred, [[2,2,1,1]])
        np.testing.assert_array_equal(erased, [[False,False,True,True]])
        self.assertTrue((out[0,:2] == 0).all())
        self.assertTrue((out[0,2:] == 255).all())
        self.assertTrue((rgb == 0).all())

    def test_unified_uses_same_policy(self):
        rgb = np.zeros((1,3,3),np.uint8)
        labels = np.array([[0,1,2]],np.uint8)
        pred, erased, _ = fuse_and_erase(rgb, [labels])
        np.testing.assert_array_equal(pred, labels)
        np.testing.assert_array_equal(erased, [[False,True,False]])

    def test_erasure_and_ignore(self):
        target = np.array([[1,1,2,0,255]],np.uint8)
        pred = np.array([[1,0,1,0,1]],np.uint8)
        erased = pred == 1
        result = summarize([counts(pred,erased,target)])
        self.assertEqual(result['arrow_residual_rate'], .5)
        self.assertEqual(result['bond_false_erasure_rate'], 1.)
        self.assertAlmostEqual(result['arrow_iou'], 1/3)
        self.assertEqual(sum(map(sum,result['confusion'])),4)

    def test_absence_is_not_perfect(self):
        target = np.zeros((1,2),np.uint8)
        result = summarize([counts(target,np.zeros_like(target,dtype=bool),target)])
        self.assertIsNone(result['arrow_iou'])
        self.assertIsNone(result['arrow_residual_rate'])
        self.assertIsNone(result['bond_false_erasure_rate'])

    def test_pool_counts_not_image_means(self):
        a = np.ones((1,1),np.uint8)
        b = np.ones((1,3),np.uint8)
        result = summarize([counts(a,a.astype(bool),a),counts(b,np.zeros_like(b,dtype=bool),b)])
        self.assertEqual(result['arrow_residual_rate'],.75)


if __name__ == '__main__':
    unittest.main()
