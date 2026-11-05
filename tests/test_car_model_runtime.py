import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import car_model_runtime as runtime


class CarModelRuntimeSafety(unittest.TestCase):
    def row(self):
        return dict(make='Toyota', model_std='Camry', year='2021', km_mileage='10000',
                    trim_std='', city='', transmission='', type_fuel='', condition='used')

    def test_price_rejects_unseen_make_model_pair(self):
        with self.assertRaisesRegex(ValueError, 'غير مدعومة'):
            runtime.validate_price({**self.row(), 'make': 'BMW'}, {'Toyota|Camry'})

    def test_price_rejects_missing_condition_and_mileage(self):
        for key in ['condition', 'km_mileage']:
            row = self.row(); del row[key]
            with self.assertRaisesRegex(ValueError, 'حقل مطلوب'):
                runtime.validate_price(row, {'Toyota|Camry'})

    def test_price_rejects_new_and_invalid_numerics(self):
        for change in [{'condition': 'new'}, {'year': '2021.5'}, {'year': True},
                       {'year': None}, {'km_mileage': 'nan'}, {'km_mileage': -1},
                       {'km_mileage': float('inf')}, {'km_mileage': ''}]:
            with self.subTest(change=change), self.assertRaises(ValueError):
                runtime.validate_price({**self.row(), **change}, {'Toyota|Camry'})

    def test_price_passes_only_allowed_features(self):
        row = {**self.row(), 'flag_bad': True, 'status': 'bad', 'sar_price': 9, 'group_id': 'g'}
        clean = runtime.validate_price(row, {'Toyota|Camry'})
        self.assertEqual(set(clean), set(runtime.FEATURES))
        self.assertEqual(clean['trim_std'], '')

    def test_search_excludes_listing_from_ineligible_query_photo(self):
        def row(uid, group, sha):
            return dict(listing_uid=uid, group_id=group, sha256=sha, relative_path=sha)
        known = [row('self', 'g0', 'interior')]
        gallery = [row('self', 'g0', 'exterior'), row('copy', 'g0', 'other'),
                   row('ok', 'g1', 'ok'), row('ok', 'g2', 'second'), row('again', 'g3', 'ok'),
                   row('end', 'g4', 'end')]
        result = runtime.ranked_gallery(gallery, np.arange(6)[::-1], 'interior', known, 5)
        self.assertEqual([r['معرف_الإعلان'] for r in result], ['ok', 'end'])

    def test_search_rejects_invalid_k(self):
        for k in [0, 21, True, 1.5]:
            with self.assertRaises(ValueError):
                runtime.ranked_gallery([], np.array([]), 'x', [], k)

    def test_inconsistent_brand_model_cannot_be_approved(self):
        make = {'مقبول': True, 'النتيجة': 'Toyota'}
        model = {'مقبول': True, 'النتيجة': 'BMW|7 Series'}
        result = runtime.recognition_result(make, model)
        self.assertEqual(result['المسار'], 'make_only')
        self.assertEqual(result['الموديل_المعتمد'], 'غير متأكد')

    def test_low_confidence_model_does_not_suppress_valid_make(self):
        result = runtime.recognition_result({'مقبول': True, 'النتيجة': 'Toyota'},
                                            {'مقبول': False, 'النتيجة': 'غير متأكد'})
        self.assertEqual(result['المسار'], 'make_only')

    def test_no_valid_make_routes_to_unavailable(self):
        result = runtime.recognition_result({'مقبول': False, 'النتيجة': 'غير متأكد'},
                                            {'مقبول': True, 'النتيجة': 'Toyota|Camry'})
        self.assertEqual(result['المسار'], 'unavailable')

    def test_locked_artifact_change_stops_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); (root / 'model.pt').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'تغيّر'):
                runtime.verify_locked_files(root, {'artifacts_sha256': {'model.pt': 'wrong'}, 'source_code_sha256': {}})

    def test_invalid_image_is_rejected_before_forward(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'image.jpg'; path.write_text('invalid')
            instance = object.__new__(runtime.CarModelRuntime)
            with self.assertRaisesRegex(ValueError, 'غير صالحة'):
                instance._image(path)


if __name__ == '__main__':
    unittest.main()
