import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from brand_final_round import weights_for
from brand_final_finish import grouped_intervals


class GroupContributionTests(unittest.TestCase):
    def test_group_bootstrap_preserves_rare_class_and_paired_identity(self):
        rows=[{'group_id':'a'}]*8+[{'group_id':'b'}]*3
        y=np.asarray(['A']*8+['B']*3)
        ci,samples=grouped_intervals(rows,y,{'one':y,'same':y.copy()},['A','B'])
        self.assertEqual(ci['one']['macro_F1'],[1.,1.])
        np.testing.assert_array_equal(samples['one']-samples['same'],np.zeros(2000))

    def test_many_images_do_not_multiply_group_contribution(self):
        rows=[{'group_id':'a','label_make':'A'}]*10+[{'group_id':'b','label_make':'A'}]
        weights=weights_for(rows,False)
        self.assertAlmostEqual(weights[:10].sum(),weights[10])
        self.assertAlmostEqual(weights.mean(),1.)

    def test_balancing_uses_distinct_groups_not_image_counts(self):
        rows=[{'group_id':'a','label_make':'A'}]*10+[{'group_id':'b','label_make':'B'},{'group_id':'c','label_make':'B'}]
        weights=weights_for(rows,True)
        self.assertAlmostEqual(weights[:10].sum()/weights[10],np.sqrt(2))
        self.assertAlmostEqual(weights[10],weights[11])


if __name__=='__main__':unittest.main()
