import unittest
import numpy as np
import torch
from core import ArrowDataset, Metrics, collapse, loss_fn
from pathlib import Path

class Tests(unittest.TestCase):
    def test_mapping(self):
        np.testing.assert_array_equal(collapse(np.array([0,1,2,3,255])),[0,1,1,2,255])

    def test_erasure_and_ignore(self):
        metric = Metrics()
        metric.update(np.array([[1,1,2,0]]),np.array([[2,3,0,255]]),4)
        r = metric.result()
        self.assertEqual(r['arrow_recall'],1.)
        self.assertEqual(r['bond_false_erasure_rate'],1.)
        self.assertAlmostEqual(r['arrow_iou'],1/3)
        self.assertIsNone(r['negative_image_false_positive_rate'])

    def test_ignored_loss(self):
        for n in [3,4]:
            x = torch.randn(2,n,8,8,requires_grad=True)
            y = torch.full((2,8,8),255,dtype=torch.long)
            loss = loss_fn(x,y)
            self.assertEqual(loss.item(),0.)
            loss.backward()
            self.assertTrue(torch.isfinite(x.grad).all())

    def test_identical_inputs(self):
        root = Path(__file__).resolve().parent/'dataset/UnifiedArrow-Type-Seg-v1'
        b,c = [ArrowDataset(root,'train',n,size=64,epoch_samples=10) for n in [3,4]]
        for epoch in [0,1]:
            b.epoch=c.epoch=epoch
            for i in range(10):
                x,y,_,sid = b[i]
                xx,yy,_,ss = c[i]
                self.assertEqual(sid,ss)
                self.assertTrue(torch.equal(x,xx))
                np.testing.assert_array_equal(y.numpy(),collapse(yy.numpy()))

if __name__=='__main__':
    unittest.main()
