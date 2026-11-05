import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import unittest
import numpy as np
from PIL import Image
from ml_visual import aggregate_groups, color_features, search_vectors, select_rejection_threshold


class VisualSafety(unittest.TestCase):
    def test_aggregation_does_not_merge_distinct_listings_in_leakage_cluster(self):
        rows=[{'group_id':'cluster','listing_uid':'ad1','label':'A'},
              {'group_id':'cluster','listing_uid':'ad1','label':'A'},
              {'group_id':'cluster','listing_uid':'ad2','label':'B'}]
        y,p,ids=aggregate_groups(rows,np.array([[.8,.2],[.6,.4],[.1,.9]]),'label')
        self.assertEqual(list(y),['A','B']); self.assertEqual(ids,['ad1','ad2'])
        np.testing.assert_allclose(p,[[.7,.3],[.1,.9]])

    def test_search_excludes_self_hash_group_and_repeated_groups(self):
        rows=[{'group_id':'self','sha256':'one'},{'group_id':'other','sha256':'selfhash'},
              {'group_id':'ok','sha256':'three'},{'group_id':'ok','sha256':'four'},
              {'group_id':'last','sha256':'five'}]
        e=np.array([[1.,0],[.99,0],[.9,0],[.8,0],[.7,0]])
        got=search_vectors(np.array([1.,0]),e,rows,k=5,exclude_group='self',exclude_sha='selfhash')
        self.assertEqual([r['group_id'] for r in got],['ok','last'])

    def test_color_statistics_are_finite_fixed_length(self):
        f=color_features(Image.new('RGB',(400,300),'red'))
        self.assertEqual(f.shape,(124,)); self.assertTrue(np.isfinite(f).all())

    def test_rejection_fails_closed_without_validation_support(self):
        p=np.tile([.51,.49],(25,1)); y=np.array(['B']*25)
        result=select_rejection_threshold(y,p,['A','B'])
        self.assertGreater(result['threshold'],1); self.assertEqual(result['validation_coverage'],0)


if __name__=='__main__': unittest.main()
