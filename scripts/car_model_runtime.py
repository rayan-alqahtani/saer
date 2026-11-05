"""واجهة استدلال محلية للنماذج المجمدة؛ لا تدريب أو اختيار أو قراءة لمقاييس test."""
from pathlib import Path
import argparse
import json
import math
import joblib
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError
import torch
from torchvision.models import ResNet18_Weights

import improve_common as protocol
from improve_visual import Tail, baseline_probs, route_probs
from ml_visual import color_features
from ml_price import FEATURES


def verify_locked_files(root, lock):
    for section, base in [('artifacts_sha256', root), ('source_code_sha256', protocol.ROOT)]:
        for name, digest in lock[section].items():
            path = base / name
            if not path.is_file() or protocol.sha256(path) != digest:
                raise ValueError('ملف النموذج أو كود التقييم تغيّر: ' + name)


def validate_price(row, supported):
    if not isinstance(row, dict):
        raise ValueError('مدخل السعر يجب أن يكون كائن حقول')
    for key in FEATURES + ['condition']:
        if key not in row:
            raise ValueError('حقل مطلوب: ' + key)
    if row['condition'] != 'used':
        raise ValueError('نموذج السعر مخصص للمستعمل فقط')
    for key in ['make', 'model_std', 'trim_std', 'city', 'transmission', 'type_fuel']:
        if not isinstance(row[key], str):
            raise ValueError('الحقل النصي غير صالح: ' + key)
    if not row['make'].strip() or not row['model_std'].strip():
        raise ValueError('حدد الماركة والموديل')
    if row['make'] + '|' + row['model_std'] not in supported:
        raise ValueError('تركيبة الماركة والموديل غير مدعومة في تدريب السعر')
    try:
        if isinstance(row['year'], bool) or isinstance(row['km_mileage'], bool):
            raise ValueError()
        year, km = float(row['year']), float(row['km_mileage'])
        if not math.isfinite(year) or not year.is_integer() or not 1950 <= year <= 2027:
            raise ValueError()
        if not math.isfinite(km) or km < 0:
            raise ValueError()
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError('سنة أو ممشى غير صالح؛ لا تعويض للقيم الناقصة') from exc
    # تُمرر القائمة البيضاء وحدها؛ لا تدخل الأعلام أو المعرفات أو الهدف.
    return {key: row[key] for key in FEATURES}


def ranked_gallery(rows, scores, digest, known_rows, k):
    if isinstance(k, bool) or not isinstance(k, int) or not 1 <= k <= 20:
        raise ValueError('عدد نتائج البحث من 1 إلى 20')
    matched = [r for r in known_rows if r['sha256'] == digest]
    excluded_groups = {r['group_id'] for r in matched}
    excluded_listings = {r['listing_uid'] for r in matched}
    seen_groups, seen_sha, seen_listings, result = set(), set(), set(), []
    for i in np.argsort(-scores, kind='stable'):
        r = rows[i]
        if (r['sha256'] == digest or r['group_id'] in excluded_groups
                or r['listing_uid'] in excluded_listings or r['group_id'] in seen_groups
                or r['sha256'] in seen_sha or r['listing_uid'] in seen_listings):
            continue
        seen_groups.add(r['group_id']); seen_sha.add(r['sha256']); seen_listings.add(r['listing_uid'])
        result.append({'معرف_الإعلان': r['listing_uid'], 'الصورة': r['relative_path'],
                       'التشابه': float(scores[i])})
        if len(result) == k:
            break
    return result


def recognition_result(make, model):
    matched = (make['مقبول'] and model['مقبول']
               and model['النتيجة'].split('|', 1)[0] == make['النتيجة'])
    return {'الماركة': make, 'الموديل': model,
            'المسار': 'model_and_make' if matched else 'make_only' if make['مقبول'] else 'unavailable',
            'الموديل_المعتمد': model['النتيجة'] if matched else 'غير متأكد',
            'نطاق_الموديل': '59 فئة؛ لا تحقق من السيارات خارج النطاق'}


class CarModelRuntime:
    def __init__(self):
        torch.set_num_threads(4)
        protocol.guard()
        self.root = protocol.session()
        self.lock = protocol.load_json(self.root / 'selection_lock.json')
        verify_locked_files(self.root, self.lock)
        identity = protocol.load_json(self.root / 'protocol.json')
        check, _ = protocol.compare_inputs()
        for name, actual in check['files'].items():
            if actual['sha256'] != identity['input_files'][name]['sha256']:
                raise ValueError('تغير ملف البيانات المعتمد: ' + name)
        self.config = {t: protocol.load_json(self.root / t / 'selected.json')
                       for t in ['make', 'model', 'color', 'body', 'price', 'search']}
        self.models = {}
        self.transform = ResNet18_Weights.IMAGENET1K_V1.transforms()
        self.price_model = joblib.load(self.root / 'price/best_model.joblib')
        self.head = joblib.load(self.root / 'make/adapted_head.joblib')
        self.gallery = None

    def _tail(self, task):
        if task not in self.models:
            cfg = self.config[task]
            model = Tail(task)
            # الفئات المحفوظة تحتوي numpy.str_؛ اسمح بهذه الأنواع المحددة فقط.
            with torch.serialization.safe_globals([np._core.multiarray.scalar, np.dtype, np.dtypes.StrDType]):
                saved = torch.load(self.root / cfg['checkpoint'], map_location='cpu', weights_only=True)
            model.load_state_dict(saved['state_dict'])
            self.models[task] = model.eval()
        return self.models[task]

    def _image(self, path):
        if path is None:
            raise ValueError('حدد مسار صورة الإدخال')
        path = Path(path)
        if not path.is_file() or path.stat().st_size > 15 * 1024 * 1024:
            raise ValueError('الصورة مفقودة أو حجمها يتجاوز 15 ميجابايت')
        try:
            with Image.open(path) as src:
                if src.format not in {'JPEG', 'PNG', 'WEBP'} or src.width * src.height > 40_000_000:
                    raise ValueError('الصورة يجب أن تكون JPG أو PNG أو WebP وبأبعاد مسموحة')
                im = ImageOps.exif_transpose(src).convert('RGB')
                return self.transform(im)[None], torch.tensor(color_features(im)[None])
        except (OSError, UnidentifiedImageError, Image.DecompressionBombError) as exc:
            raise ValueError('الصورة غير صالحة') from exc

    def _forward(self, task, path, inputs):
        cfg = self.config[task]
        if cfg['kind'] != 'tail':
            p, classes = baseline_probs(task, [{'external_path': str(Path(path).resolve())}])
            return p, classes, None
        x, extra = inputs
        with torch.inference_mode():
            model = self._tail(task)
            logits, features = model(x, extra, 0, True)
            p = logits.softmax(1)
            if cfg.get('tta'):
                p = (p + model(x.flip(-1), extra, 0).softmax(1)) / 2
        return p.numpy(), cfg['classes'], features.numpy()

    def _make(self, path, inputs, model_output):
        cfg = self.config['make']
        pc, labels, features = model_output
        if cfg.get('independent_source') == 'adapted_head':
            if features is None:
                raise ValueError('تمثيل الماركة المحسن غير متوفر')
            pm = self.head['model'].predict_proba(self.head['scaler'].transform(features))
            classes = self.head['classes']
        else:
            pm, classes = baseline_probs('make', [{'external_path': str(Path(path).resolve())}])
        if cfg['kind'] == 'independent':
            return pm, classes, pm.max(1)
        if cfg['model_source'] != 'improved':
            pc, labels = baseline_probs('model', [{'external_path': str(Path(path).resolve())}])
        p = route_probs(pm, pc, labels, classes, cfg['threshold'], cfg['method'])
        confidence = np.where(pc.max(1) >= cfg['threshold'], pc.max(1), pm.max(1))
        return p, classes, confidence

    def _result(self, task, p, classes, confidence=None):
        threshold = self.config[task].get('rejection', {}).get('threshold', 1.01)
        confidence = float(p.max()) if confidence is None else float(confidence[0])
        order = np.argsort(-p[0])[:3]
        accepted = confidence >= threshold
        return {'الحالة': 'تجريبي', 'النتيجة': classes[order[0]] if accepted else 'غير متأكد',
                'مقبول': bool(accepted), 'درجة_القرار': confidence, 'عتبة_القبول': threshold,
                'اقتراحات': [{'الفئة': classes[i], 'درجة_قرار_غير_مضمونة': float(p[0, i])} for i in order],
                'تنبيه': 'درجات القرار ليست احتمالات صحة معايرة؛ التعميم خارج المصدر لم يختبر'}

    def price(self, row):
        clean = validate_price(row, self.price_model.features.model_counts)
        value = float(self.price_model.predict([clean])[0])
        if not math.isfinite(value) or value <= 0:
            raise ValueError('تعذر إنتاج سعر صالح')
        return {'سعر_الإعلان_المقدر_ريال': round(value, 2), 'الحالة': 'تجريبي',
                'تنبيه': 'بيانات السيارة يقدمها المستخدم؛ سعر إعلان للمستعمل وليس سعر بيع مضمونًا أو نطاق سعر معايرًا'}

    def _search(self, path, features, k):
        if self.config['search']['kind'] != 'finetuned_cosine' or features is None:
            raise ValueError('نوع فهرس البحث غير مدعوم بهذه الواجهة')
        if self.gallery is None:
            rows = protocol.compact_rows('search', 'train')
            index = protocol.read_csv(self.root / 'search/finetuned_index.csv')
            lookup = {r['sha256']: i for i, r in enumerate(index)}
            array = np.load(self.root / 'search/finetuned_features.npy', mmap_mode='r')
            vectors = np.asarray(array[[lookup[r['sha256']] for r in rows]])
            known = protocol.read_csv(protocol.ROOT / 'الملفات_المهمة/ربط_الصور.csv')
            self.gallery = rows, vectors, known
        rows, vectors, known = self.gallery
        results = ranked_gallery(rows, vectors @ features[0], protocol.sha256(path), known, k)
        return {'الحالة': 'تجريبي', 'النتائج': results,
                'تنبيه': 'معرض التدريب فقط؛ استبعاد الإعلان والمجموعة والبصمة من كامل جدول الصور. التشابه ليس احتمال صحة'}

    def predict(self, task, path=None, row=None, k=5):
        if task == 'price':
            return self.price(row)
        if task == 'status':
            return {'الحالة': 'تجريبي', 'الجلسة': self.root.name,
                    'سلامة_النماذج': True, 'التقييم_الخارجي': 'لم يختبر بعد',
                    'المهام': ['price', 'recognize', 'make', 'model', 'color', 'body', 'search']}
        if task not in {'recognize', 'make', 'model', 'color', 'body', 'search'}:
            raise ValueError('مهمة غير مدعومة')
        inputs = self._image(path)
        if task in {'color', 'body'}:
            p, classes, _ = self._forward(task, path, inputs)
            return self._result(task, p, classes)
        output = self._forward('model', path, inputs)
        if task == 'search':
            return self._search(path, output[2], k)
        if task == 'model':
            return self._result('model', output[0], output[1])
        p, classes, conf = self._make(path, inputs, output)
        make = self._result('make', p, classes, conf)
        return make if task == 'make' else recognition_result(make, self._result('model', output[0], output[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('task', choices=['status', 'price', 'recognize', 'make', 'model', 'color', 'body', 'search'])
    parser.add_argument('--input', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--k', type=int, default=5)
    args = parser.parse_args()
    try:
        runtime = CarModelRuntime()
        row = protocol.load_json(args.input) if args.task == 'price' and args.input else None
        payload = {'نجاح': True, 'النتيجة': runtime.predict(args.task, args.input, row, args.k)}
        code = 0
    except (ValueError, TypeError, KeyError, OSError, RuntimeError) as exc:
        payload = {'نجاح': False, 'الخطأ': str(exc)}
        code = 1
    if args.output:
        protocol.save_json(args.output, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return code


if __name__ == '__main__':
    raise SystemExit(main())
