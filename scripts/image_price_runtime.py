"""استدلال صورة ← سعر؛ لا يعتمد على قاعدة إعلانات أو بحث أقرب جار."""
import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import torch
from PIL import Image, ImageOps, UnidentifiedImageError
from torchvision.models import resnet18, ResNet18_Weights


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def load_image(path):
    try:
        with Image.open(path) as source:
            if source.width * source.height > 40_000_000:
                raise ValueError('الصورة كبيرة جدًا؛ الحد 40 مليون بكسل')
            if min(source.size) < 128:
                raise ValueError('الصورة صغيرة جدًا؛ أقصر ضلع يجب أن يكون 128 بكسل على الأقل')
            if source.format not in {'JPEG', 'PNG', 'WEBP'}:
                raise ValueError('الصيغ المدعومة JPEG وPNG وWebP')
            return ImageOps.exif_transpose(source).convert('RGB')
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError('تعذر قراءة الصورة') from exc


def domain_score(detector, features):
    z = detector['pca'].transform(features)
    reconstructed = detector['pca'].inverse_transform(z)
    residual = np.mean((features - reconstructed) ** 2, axis=1)
    distance = np.mean(z ** 2 / np.maximum(detector['pca'].explained_variance_, 1e-8), axis=1)
    return np.maximum(residual / detector['residual_scale'], distance / detector['distance_scale'])


class ImagePriceRuntime:
    def __init__(self, bundle):
        self.bundle = Path(bundle)
        self.policy = json.loads((self.bundle / 'runtime_policy.json').read_text(encoding='utf-8'))
        for name, expected in self.policy['artifacts_sha256'].items():
            if digest(self.bundle / name) != expected:
                raise ValueError('بصمة ملف النموذج غير مطابقة: ' + name)
        torch.set_num_threads(4)
        self.encoder = resnet18(weights=None)
        self.encoder.load_state_dict(torch.load(self.bundle / 'encoder_weights.pth', map_location='cpu', weights_only=True))
        self.encoder.fc = torch.nn.Identity()
        self.encoder.eval()
        self.transform = ResNet18_Weights.IMAGENET1K_V1.transforms()
        self.regressor = joblib.load(self.bundle / 'regressor.joblib') if self.policy['kind'] != 'tail' else None
        self.domain = joblib.load(self.bundle / 'domain.joblib')
        self.tail = None
        if self.policy['kind'] == 'tail':
            self.tail = resnet18(weights=None)
            self.tail.fc = torch.nn.Linear(512, 1)
            self.tail.load_state_dict(torch.load(self.bundle / 'price_network.pt', map_location='cpu', weights_only=True))
            self.tail.eval()

    def predict(self, image_path):
        im = load_image(image_path)
        x = self.transform(im)[None]
        with torch.inference_mode():
            feature = self.encoder(x).numpy()
            feature /= np.maximum(np.linalg.norm(feature, axis=1, keepdims=True), 1e-12)
            if self.tail is not None:
                b = self.tail
                z = b.maxpool(b.relu(b.bn1(b.conv1(x))))
                z = b.layer4(b.layer3(b.layer2(b.layer1(z))))
                z = torch.nn.functional.normalize(b.avgpool(z).flatten(1), dim=1)
                value = b.fc(z).numpy().reshape(-1)
            else:
                value = self.regressor.predict(feature)
        score = float(domain_score(self.domain, feature)[0])
        if not np.isfinite(value[0]):
            raise ValueError('تعذر إنتاج سعر صالح من الصورة')
        accepted = bool(np.isfinite(score) and score <= self.policy['domain_threshold'])
        price = float(np.exp(np.clip(value[0], *self.policy['log_prediction_bounds'])))
        return {
            'status': 'experimental' if accepted else 'abstained',
            'predicted_asking_price_sar': round(price, 2) if accepted else None,
            'message_ar': 'سعر إعلان تقديري من الصورة وحدها' if accepted else 'تعذر التقدير: الصورة بعيدة عن توزيع صور التدريب',
            'domain_score': score, 'domain_threshold': self.policy['domain_threshold'],
            'domain_check_is_heuristic': True, 'calibrated_confidence': None,
            'limitations_ar': ['تقدير تجريبي؛ لا يعرف الممشى أو الحالة الميكانيكية',
                               'فحص النطاق لا يضمن كشف سيارة غير مدعومة أو صورة غير سيارة',
                               'لم يُختبر على صور خارج المصدر؛ لا يوجد نطاق سعر معاير'],
            'inference_uses_listing_database': False,
            'price_scope_ar': 'يفترض التقدير أن السيارة مستعملة؛ لا يحدد حالتها من الصورة',
        }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--bundle', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--image', type=Path, required=True)
    args = parser.parse_args()
    try:
        print(json.dumps(ImagePriceRuntime(args.bundle).predict(args.image), ensure_ascii=False, indent=2))
    except (ValueError, FileNotFoundError) as exc:
        print(json.dumps({'status': 'invalid_input', 'message_ar': str(exc)}, ensure_ascii=False))
        raise SystemExit(2)
