import sys,unittest,tempfile,json
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import numpy as np
import improve_common as protocol
from improve_visual import route_probs
from improve_price import PriceCandidate


class ImprovementSafety(unittest.TestCase):
    def test_route_uses_model_prediction_with_independent_fallback(self):
        p=np.array([[.8,.2],[.8,.2]]);m=np.array([[.01,.99],[.51,.49]])
        got=route_probs(p,m,['A|one','B|two'],['A','B'],.8,'hard')
        self.assertEqual(got.argmax(1).tolist(),[1,0])

    def test_bootstrap_resamples_complete_groups(self):
        rows=[{'group_id':'g'}]*10+[{'group_id':'h'}]*10
        y=['A']*20;pred=['A']*10+['B']*10
        result,accuracy,f1=protocol.class_bootstrap(rows,y,pred,['A','B'],reps=200)
        self.assertEqual(result['groups'],2)
        self.assertTrue(set(np.unique(accuracy)).issubset({0.,.5,1.}))

    def test_second_test_acquisition_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'model').mkdir();(root/'selection_lock.json').write_text(json.dumps({'artifacts_sha256':{}}))
            with patch.object(protocol,'session',return_value=root),patch.object(protocol,'guard'):
                protocol.acquire_test('model')
                with self.assertRaises(FileExistsError):protocol.acquire_test('model')

    def test_experiments_cannot_change_after_final_lock(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'selection_lock.json').write_text('{}')
            with patch.object(protocol,'session',return_value=root):
                with self.assertRaises(ValueError):protocol.experiment('model','late','change',.4,.5,True)

    def test_price_flags_and_identifiers_cannot_affect_features(self):
        rows=[{'make':'Toyota','model_std':'Camry','year':str(2018+i%4),'km_mileage':str(10000+i*1000),
               'trim_std':'LE','city':'Riyadh','transmission':'Automatic','type_fuel':'Gasoline','sar_price':str(40000+i*500),'group_id':str(i)} for i in range(24)]
        model=PriceCandidate({'kind':'hgb','loss':'absolute_error','max_iter':5,'min_samples_leaf':2}).fit(rows)
        changed=[{**r,'sar_price':'9999999','flag_source_unavailable':'True','price_status':'other','group_id':'changed','listing_uid':'other'} for r in rows]
        np.testing.assert_allclose(model.predict(rows),model.predict(changed))

    def test_guard_rejects_changed_split_even_if_current_manifest_changed(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            with patch.object(protocol,'session',return_value=root),patch.object(protocol,'save_json'), \
                 patch.object(protocol,'compare_inputs',return_value=({'critical':False,'split_fingerprint':'new'},{})), \
                 patch.object(protocol,'load_json',return_value={'split_fingerprint':'old','original_models_sha256':{}}):
                with self.assertRaises(ValueError):protocol.guard()


if __name__=='__main__':unittest.main()
