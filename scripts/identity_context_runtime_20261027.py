"""Single-image identity candidate, frozen before external diagnostics."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import joblib
from scipy.special import softmax
from model_image_tester_20261026 import CandidateRuntime
import visual_fusion_predict as base

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / 'models/identity_context_20261027_v1'


def apply_policy(probs, classes, policy):
    calibrated = {t: softmax(np.log(np.maximum(p, 1e-10)) / policy['temperatures'][t], axis=1)
                  for t, p in probs.items()}
    logits = np.log(np.maximum(calibrated['model'], 1e-10))
    for task, power in [('make', policy['choice']['make_weight']), ('body', policy['choice']['body_weight'])]:
        if power == 0:
            continue
        p = calibrated[task]
        reliability = np.clip(1. + (p * np.log(np.maximum(p, 1e-10))).sum(axis=1) / np.log(p.shape[1]), 0, 1)
        if task == 'make':
            evidence = p[:, [classes['make'].index(c.split('|', 1)[0]) for c in classes['model']]]
        else:
            evidence = p @ np.asarray(policy['body_compatibility']).T
        logits += power * reliability[:, None] * np.log(np.maximum(evidence, 1e-8))
    return softmax(logits, axis=1)


class IdentityContextRuntime(CandidateRuntime):
    def __init__(self):
        super().__init__()
        self.release = json.loads((BUNDLE / 'release.json').read_text(encoding='utf-8'))
        from improve_common import sha256
        for name, digest in self.release['sha256'].items():
            if sha256(ROOT / name) != digest:
                raise RuntimeError('تغيّر ملف من ملفات المرشح: ' + name)
        self.policy = json.loads((BUNDLE / 'selection_lock.json').read_text(encoding='utf-8'))
        self.context_classes = json.loads((BUNDLE / 'classes.json').read_text(encoding='utf-8'))
        self.direct_heads = {t: joblib.load(base.train.RUN / t / 'head.joblib') for t in ('model', 'body')}

    def context_probabilities(self, features):
        probs = {}
        make = self.artifacts['make']
        x = self.candidate_features('make', features)
        probs['make'] = make['model'].predict_proba(make['scaler'].transform(x))
        assert list(make['classes']) == self.context_classes['make']
        tasks = ['model'] + (['body'] if self.policy['choice']['body_weight'] else [])
        for task in tasks:
            art = self.direct_heads[task]
            assert list(art['classes']) == self.context_classes[task]
            x = base.features(features, art['kind'], task)
            probs[task] = art['model'].predict_proba(art['scaler'].transform(x))
        return apply_policy(probs, self.context_classes, self.policy), self.context_classes['model']

    def predict(self, task, path, k=5, features=None):
        if task != 'recognize':
            return super().predict(task, path, k, features)
        f = features if features is not None else self.extract(path)
        original = super().predict(task, path, k, f)
        p, classes = self.context_probabilities(f)
        top = self._top(p[0], classes)
        label = top[0]['الفئة']
        make, _, model = label.partition('|')
        original['الموديل_قبل_الترجيح'] = original['الاقتراح_الرئيسي']['النتيجة']
        original['الموديل_المستقل'] = {'النتيجة': label, 'أفضل_الاحتمالات': top}
        original['الاقتراح_الرئيسي'] = {'النتيجة': label, 'الماركة': make, 'الموديل': model,
                                      'درجة_غير_معايرة': top[0]['درجة_غير_معايرة'], 'أفضل_الاحتمالات': top}
        original['اتساق_الماركة_والموديل'] = make == original['الماركة_المستقلة']['النتيجة']
        original['الحالة'] = 'مرشح تطويري — ترجيح الهوية'
        original['المسار'] = 'DINOv2 للموديل مع ترجيح تدريجي من رأس الماركة المطوّر؛ لا استبعاد إلزامي'
        original['نطاق_الموديل'] = 'الفئات الـ59 فقط؛ المعايرة داخلية ولا تثبت الصحة خارج المصدر'
        original['يلزم_تأكيد_المستخدم_قبل_التسعير'] = True
        return original


def main():
    import tkinter as tk
    from model_image_tester import ModelImageTester
    root = tk.Tk()
    root.title('سعّر — تجربة ترجيح هوية السيارة')
    ModelImageTester(root, engine_factory=IdentityContextRuntime,
        subtitle='مرشح 27 أكتوبر — ترجيح الموديل بإشارة الماركة؛ يلزم تأكيد السيارة قبل التسعير')
    root.mainloop()


if __name__ == '__main__':
    main()
