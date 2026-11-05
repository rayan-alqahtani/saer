import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import unittest
import numpy as np
from ml_price import PriceFeatures, PriceModel, FEATURES
from ml_common import assert_split_integrity


class PriceSafety(unittest.TestCase):
    def setUp(self):
        self.rows=[{'make':'Toyota','model_std':'Camry','year':str(2018+i%5),'km_mileage':str(10000+i*1000),
                    'trim_std':'LE','city':'Riyadh','transmission':'Automatic','type_fuel':'Gasoline',
                    'sar_price':str(50000+i*1000),'group_id':str(i),'split':'train'} for i in range(30)]

    def test_forbidden_columns_do_not_affect_prediction(self):
        m=PriceModel('hgb',{'max_iter':10,'min_samples_leaf':3}).fit(self.rows)
        changed=[{**r,'sar_price':'99999999','flag_price_iqr_outlier':'True','group_id':'hacked','split':'test','source_dataset':'other'} for r in self.rows]
        np.testing.assert_allclose(m.predict(self.rows),m.predict(changed))

    def test_unknown_categories_are_not_fitted_at_inference(self):
        m=PriceModel('hgb',{'max_iter':10,'min_samples_leaf':3}).fit(self.rows)
        before=[list(c) for c in m.encoder.categories_]
        out=m.predict([{**self.rows[0],'make':'UNSEEN','model_std':'UNSEEN','trim_std':'UNSEEN','city':'UNSEEN'}])
        self.assertTrue(np.isfinite(out).all()); self.assertEqual(before,[list(c) for c in m.encoder.categories_])

    def test_missing_mileage_fails_instead_of_imputing(self):
        f=PriceFeatures().fit(self.rows)
        with self.assertRaises(ValueError): f.transform([{**self.rows[0],'km_mileage':''}])

    def test_cross_split_group_rejected(self):
        with self.assertRaises(ValueError): assert_split_integrity([{'group_id':'same','split':'train'},{'group_id':'same','split':'test'}])

    def test_monotonic_hgb_for_age_and_mileage(self):
        m=PriceModel('hgb',{'max_iter':15,'min_samples_leaf':3}).fit(self.rows)
        r=self.rows[0]
        pred=m.predict([{**r,'year':str(y),'km_mileage':'10000'} for y in range(2018,2023)])
        self.assertTrue((np.diff(pred)>=-1e-7).all())
        pred=m.predict([{**r,'km_mileage':str(k)} for k in [0,10000,50000,100000]])
        self.assertTrue((np.diff(pred)<=1e-7).all())


if __name__=='__main__': unittest.main()
