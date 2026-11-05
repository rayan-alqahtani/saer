"""تشغيل المرشح متعدد صور التدريب بصورة واحدة فقط، دون قاعدة إعلانات."""
import argparse
import hashlib
import json
import sys
from pathlib import Path
import numpy as np


class ImagePriceRuntime:
    def __init__(self, bundle):
        self.bundle = Path(bundle)
        lock = self.bundle / 'LOCK.json'
        if lock.exists():
            for name, sha in json.loads(lock.read_text(encoding='utf-8'))['sha256'].items():
                with (self.bundle / name).open('rb') as f: actual = hashlib.file_digest(f, 'sha256').hexdigest()
                if actual != sha: raise ValueError('بصمة ملف النموذج غير مطابقة: ' + name)
        self.policy = json.loads((self.bundle / 'policy.json').read_text(encoding='utf-8'))
        sys.path.insert(0, str(self.bundle / 'vendor'))
        import gemini_features as gf
        self.gf = gf
        self.gem = gf.ImagePriceRuntime(self.bundle / 'gemini')
        with np.load(self.bundle / 'head.npz') as h: self.head = {k: h[k] for k in h.files}
        self.head['support53'] = self.head['support53'].astype(np.float64)

    def predict_batch_paths(self, paths):
        gf = self.gf; h = self.head; gem = self.gem
        pp = [gf.preprocess_image_views(gf.load_image(p)) for p in paths]
        xc = np.stack([p[0] for p in pp]); xf = np.stack([p[1] for p in pp]); raw = np.stack([p[2] for p in pp])
        a, b, c, e = [gem.backbone.run_view(v) for v in (xc, xc[:, :, :, ::-1].copy(), xf, xf[:, :, :, ::-1].copy())]
        dom, cr, fu, x = gf.build_descriptors_from_backbone_outputs(a[0], a[1], a[2], a[3], a[4], b[2], b[3], b[4], c[2], c[3], c[4], e[2], e[3], e[4], raw, gem.vs_mean, gem.vs_std)
        zg = gem.b_tot + cr @ gem.w_crop_7x4 + fu @ gem.w_full_7x4 + x @ gem.w_53_lin + ((x @ gem.V_rec) ** 2) @ gem.lk
        zg = np.clip(zg, *gem.bounds)
        v = gf.l2_norm(np.clip((raw.astype(np.float64) - h['vs_mean']) / h['vs_std'], -5, 5))
        def adj(x):
            x[:, :-154] *= np.sqrt(1 - .1 / 1.95) / np.maximum(np.linalg.norm(x[:, :-154], axis=1, keepdims=True), 1e-12)
            x[:, -154:] = v * np.sqrt(.1 / 1.95)
            return gf.l2_norm(x)
        cr = adj(cr); x = adj(x); sim = np.clip(x @ h['support53'].T, -1, 1)
        z = h['intercept'][0] + cr @ h['w_crop'] + fu @ h['w_full'] + (sim ** 2) @ h['poly_coef'] + np.exp(-1.5 * np.maximum(2 - 2 * sim, 0)) @ h['rbf_coef']
        z = np.clip(z, *self.policy['bounds']); w = self.policy['gemini_weight']
        price = np.exp(np.clip((1 - w) * z + w * zg, *self.policy['bounds']))
        return price, gem.compute_domain_score(dom)

    def predict(self, path):
        p, s = self.predict_batch_paths([path]); accepted = bool(np.isfinite(s[0]) and s[0] <= self.gem.domain_threshold)
        if not np.isfinite(p[0]): raise ValueError('تعذر إنتاج سعر صالح')
        return {'status': 'experimental' if accepted else 'abstained',
            'predicted_asking_price_sar': round(float(p[0]), 2) if accepted else None,
            'raw_predicted_asking_price_sar': round(float(p[0]), 2), 'domain_score': float(s[0]),
            'domain_threshold': self.gem.domain_threshold, 'confidence_calibrated': False,
            'listing_database_required': False,
            'message_ar': 'تقدير تجريبي لسعر إعلان مستعمل من صورة واحدة؛ لم يُختبر خارجيًا' if accepted else 'تعذر التقدير: الصورة بعيدة عن توزيع التدريب'}


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--bundle', type=Path, default=Path(__file__).parent)
    ap.add_argument('--image', type=Path, required=True); a = ap.parse_args()
    print(json.dumps(ImagePriceRuntime(a.bundle).predict(a.image), ensure_ascii=False, indent=2))
