"""تسعير هوية يختارها أو يؤكدها المستخدم؛ لا تمرير تلقائي لتخمين الصورة."""
from __future__ import annotations
import importlib.util
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class ConfirmedIdentityPricing:
    def __init__(self):
        pointer = json.loads((ROOT / 'models/delivery_current.json').read_text(encoding='utf-8-sig'))
        self.bundle = (ROOT / pointer['run_dir']).resolve()
        if not self.bundle.is_relative_to((ROOT / 'models').resolve()):
            raise ValueError('مسار حزمة السعر غير صالح')
        spec = importlib.util.spec_from_file_location('confirmed_vehicle_price_bundle', self.bundle / 'predict.py')
        self.runtime = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.runtime)
        manifest = json.loads((self.bundle / 'integrity_manifest.json').read_text(encoding='utf-8-sig'))
        for name, digest in manifest['files_sha256'].items():
            with (self.bundle / name).open('rb') as stream:
                if hashlib.file_digest(stream, 'sha256').hexdigest() != digest:
                    raise ValueError('تغير ملف في حزمة السعر: ' + name)
        model = self.runtime.load('price')
        self.support = dict(model['vocabulary']['model_counts'])

    def catalog(self):
        result = {}
        for label in sorted(self.support):
            make, model = label.split('|', 1)
            result.setdefault(make, []).append(model)
        return result

    def predict(self, *, make, model, year, km_mileage, identity_confirmed,
                trim='', city='', transmission='', fuel=''):
        if identity_confirmed is not True:
            raise ValueError('اختر السيارة من القائمة أو أكّد هوية السيارة المقترحة من الصورة أولًا')
        if make + '|' + model not in self.support:
            raise ValueError('هذه الماركة والموديل غير مدعومين في تدريب السعر')
        if isinstance(year, bool) or isinstance(km_mileage, bool):
            raise ValueError('أدخل السنة والممشى كقيم رقمية صحيحة')
        year_value, mileage_value = float(year), float(km_mileage)
        if not math.isfinite(year_value) or not year_value.is_integer() or not 1950 <= year_value <= 2027:
            raise ValueError('سنة غير صالحة')
        if not math.isfinite(mileage_value) or mileage_value < 0:
            raise ValueError('ممشى غير صالح')
        row = {'make': make, 'model_std': model, 'year': int(year_value), 'km_mileage': mileage_value,
               'condition': 'used', 'trim_std': trim, 'city': city, 'transmission': transmission, 'type_fuel': fuel}
        price = float(self.runtime.price(row)['سعر_الإعلان_المقدر_ريال'])
        support = int(self.support[make + '|' + model])
        return {'السعر_الإعلاني_التقديري_ريال': price, 'الماركة_المؤكدة': make, 'الموديل_المؤكد': model,
                'عدد_إعلانات_تدريب_السعر_لهذا_الموديل': support,
                'دعم_الموديل_قليل': support < 5,
                'تنبيه': 'تقدير لسعر إعلان سيارة مستعملة؛ ليس سعر بيع مضمونًا ولا يفحص الحالة الميكانيكية'}
