import sys
import unittest
from pathlib import Path
import csv
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from identity_to_price_handoff_20261028 import ConfirmedIdentityPricing, ROOT


class ConfirmedPricingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = ConfirmedIdentityPricing()
        with (ROOT / 'data/processed/price_dataset.csv').open(encoding='utf-8-sig', newline='') as f:
            cls.example = next(z for z in csv.DictReader(f) if z['split'] == 'train')

    def inputs(self):
        z = self.example
        return dict(make=z['make'], model=z['model_std'], year=z['year'], km_mileage=z['km_mileage'],
                    trim=z['trim_std'], city=z['city'], transmission=z['transmission'], fuel=z['type_fuel'])

    def test_unconfirmed_prediction_is_blocked(self):
        with self.assertRaisesRegex(ValueError, 'أكّد'):
            self.engine.predict(**self.inputs(), identity_confirmed=False)

    def test_crossed_make_and_model_is_blocked(self):
        args = self.inputs(); args['make'] = 'InvalidMake'
        with self.assertRaisesRegex(ValueError, 'غير مدعومين'):
            self.engine.predict(**args, identity_confirmed=True)

    def test_confirmed_price_matches_existing_model_exactly(self):
        z = self.example
        expected = self.engine.runtime.price(z)['سعر_الإعلان_المقدر_ريال']
        actual = self.engine.predict(**self.inputs(), identity_confirmed=True)
        self.assertAlmostEqual(actual['السعر_الإعلاني_التقديري_ريال'], expected, places=8)

    def test_no_year_inferred_from_image(self):
        args = self.inputs(); args['year'] = float('nan')
        with self.assertRaisesRegex(ValueError, 'سنة'):
            self.engine.predict(**args, identity_confirmed=True)


if __name__ == '__main__':
    unittest.main()
