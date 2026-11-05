import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from cache_identity_adapter_20261028 import cache_probabilities,assert_disjoint


class GalleryBalanceTests(unittest.TestCase):
    def test_repeating_same_group_view_does_not_increase_its_vote(self):
        gallery=np.array([[1.,0.],[0.,1.]],dtype='float32')
        query=np.array([[.8,.6]],dtype='float32')
        rows=[{'group_id':'one','rec_label':'A'},{'group_id':'two','rec_label':'B'}]
        original,_=cache_probabilities(gallery,query,rows,['A','B'],'model')
        repeated=np.r_[gallery,gallery[:1],gallery[:1]]
        new,_=cache_probabilities(repeated,query,rows+[rows[0],rows[0]],['A','B'],'model')
        for beta in original:np.testing.assert_allclose(original[beta],new[beta],atol=1e-6)

    def test_class_imbalance_does_not_add_prior_mass_for_identical_evidence(self):
        gallery=np.array([[1.,0.],[1.,0.],[1.,0.]],dtype='float32')
        rows=[{'group_id':'one','rec_label':'A'},{'group_id':'two','rec_label':'A'},
              {'group_id':'three','rec_label':'B'}]
        result,_=cache_probabilities(gallery,gallery[:1],rows,['A','B'],'model')
        for p in result.values():np.testing.assert_allclose(p,[[.5,.5]],atol=1e-6)

    def test_listing_collision_is_caught_even_if_group_ids_differ(self):
        x={'group_id':'one','listing_uid':'same','sha256':'first'}
        y={'group_id':'two','listing_uid':'same','sha256':'second'}
        with self.assertRaises(AssertionError):assert_disjoint([x],[y])


if __name__=='__main__':unittest.main()
