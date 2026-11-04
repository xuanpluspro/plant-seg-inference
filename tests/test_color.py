import csv
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import numpy as np
from PIL import Image
from inference.color_classification import (ColorClassifier, classify_foreground, color_mask,
                                           color_overlay, export_color_csv)

class Dummy:
    def __init__(self): self.seen=[]
    def predict_green(self, pixels):
        self.seen.extend(pixels.tolist())
        return pixels[:,1] > pixels[:,0]

class ColorTests(unittest.TestCase):
    def test_foreground_only_and_chunk_invariance(self):
        rgb=np.array([[[0,200,0],[200,20,0],[0,250,0]],[[200,20,0],[0,200,0],[0,0,0]]],dtype=np.uint8)
        fg=np.array([[1,1,0],[1,1,0]],bool)
        a=Dummy(); labels,stats=classify_foreground(rgb,fg,a,2)
        b=Dummy(); other,otherstats=classify_foreground(rgb,fg,b,999)
        np.testing.assert_array_equal(labels,[[1,2,0],[2,1,0]])
        np.testing.assert_array_equal(labels,other)
        self.assertEqual(stats,otherstats);self.assertEqual(len(a.seen),4)
        self.assertEqual(stats['senescent_ratio'],.5)
        self.assertAlmostEqual(stats['senescent_fraction'],1/3)
    def test_empty_ratio_is_missing(self):
        d=Dummy(); labels,s=classify_foreground(np.zeros((2,2,3),np.uint8),np.zeros((2,2),bool),d)
        self.assertIsNone(s['senescent_ratio']);self.assertEqual(d.seen,[])
        self.assertEqual(s['vegetation_fraction'],0)
    def test_probability_column_and_threshold(self):
        c=ColorClassifier.__new__(ColorClassifier)
        c.features=lambda x:x;c.green_column=0;c.threshold=.5
        class Model:
            def predict_proba(self,x):return np.array([[.5,.5],[.2,.8]])
        c.model=Model()
        np.testing.assert_array_equal(c.predict_green(np.zeros((2,3))),[True,False])
    def test_invalid_probability(self):
        c=ColorClassifier.__new__(ColorClassifier);c.features=lambda x:x;c.green_column=1;c.threshold=.5
        class Model:
            def predict_proba(self,x):return np.array([[float('nan'),1]])
        c.model=Model()
        with self.assertRaises(ValueError):c.predict_green(np.zeros((1,3)))
    def test_pickle_requires_explicit_trust(self):
        with self.assertRaises(ValueError):ColorClassifier('does-not-exist.pkl')
    def test_palette_roundtrip_and_original_unchanged(self):
        labels=np.array([[0,1,2]],np.uint8);image=Image.new('RGB',(3,1),(100,100,100))
        with TemporaryDirectory() as tmp:
            p=Path(tmp)/'mask.png';color_mask(labels).save(p)
            with Image.open(p) as saved:
                self.assertEqual(saved.mode,'P');np.testing.assert_array_equal(np.asarray(saved),labels)
                np.testing.assert_array_equal(np.asarray(saved.convert('RGB'))[0],[[0,0,0],[0,180,70],[255,255,0]])
        out=color_overlay(image,labels,1)
        np.testing.assert_array_equal(np.asarray(out)[0,0],[100,100,100])
        np.testing.assert_array_equal(np.asarray(image),100)
    def test_csv_empty_ratio_and_atomic_update(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'meta').mkdir()
            _,stats=classify_foreground(np.zeros((1,1,3),np.uint8),np.zeros((1,1),bool),Dummy())
            stats['color_seconds']=0
            export_color_csv(root,{'叶.jpg':{'color_stats':stats}})
            with (root/'meta/color_metrics.csv').open(encoding='utf-8-sig') as f:rows=list(csv.DictReader(f))
            self.assertEqual(rows[0]['senescent_ratio'],'');self.assertEqual(rows[0]['image'],'叶.jpg')
            self.assertFalse(list((root/'meta').glob('*.tmp')))
    def test_shape_and_chunk_validation(self):
        with self.assertRaises(ValueError):classify_foreground(np.zeros((1,1,3),np.uint8),np.zeros((2,1)),Dummy())
        with self.assertRaises(ValueError):classify_foreground(np.zeros((1,1,3),np.uint8),np.zeros((1,1)),Dummy(),0)
