"""تحقق محلي من API بأوزان مجمدة؛ لا تدريب أو نتائج بديلة."""
from pathlib import Path
import argparse
import hashlib
import json
import math
import urllib.request
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base-url', default='http://127.0.0.1:8000')
    parser.add_argument("--output", type=Path, default=ROOT / "tmp/runtime-verification.json")
    parser.add_argument("--image", type=Path, default=None, help="Locally supplied image with permission to use")
    args = parser.parse_args()
    assert urlsplit(args.base_url).hostname in {'127.0.0.1', 'localhost'}
    image = args.image.resolve() if args.image else None
    payload = dict(make='Toyota', model_std='Camry', year=2020, km_mileage=135685,
                   trim_std='LE', city='Dammam', transmission='Automatic', type_fuel='Gasoline', condition='used')
    def request(path, data=None, content_type=None):
        req = urllib.request.Request(args.base_url + path, data=data,
            headers={'Content-Type': content_type} if content_type else {})
        with urllib.request.urlopen(req, timeout=600) as response:
            return json.load(response)
    weights = json.loads((ROOT / 'docs/runtime-assets.json').read_text(encoding='utf-8'))['models']
    for item in weights:
        data = (ROOT / item['path']).read_bytes()
        assert len(data) == item['bytes'] and hashlib.sha256(data).hexdigest() == item['sha256']
    results = {'weights_verified': weights, 'image_sha256': hashlib.sha256(image.read_bytes()).hexdigest() if image else None,
               'input': payload, 'training_performed': False, 'public_encoders_downloaded': False,
               'health': request('/api/health')}
    results['tabular'] = request('/api/estimate-tabular', json.dumps(payload).encode(), 'application/json')
    assert math.isfinite(results['tabular']['سعر_الإعلان_المقدر_ريال'])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding='utf-8')
    if image is None:
        print('Tabular prediction passed. Supply --image for the two image endpoints.', flush=True)
        return
    if not image.is_file():
        raise FileNotFoundError(image)
    boundary = 'saer-runtime-check'
    body = (f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="sample.webp"\r\nContent-Type: image/webp\r\n\r\n'.encode()
            + image.read_bytes() + f'\r\n--{boundary}--\r\n'.encode())
    for key, endpoint in [('image_price', '/api/estimate-image-price'), ('identity', '/api/recognize')]:
        results[key] = request(endpoint, body, f'multipart/form-data; boundary={boundary}')
        assert results[key].get('من_ذاكرة_مؤقتة') is False, 'Expected a fresh model call; use a fresh runtime cache.'
        args.output.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding='utf-8')
        print(key, json.dumps(results[key], ensure_ascii=False), flush=True)
    assert math.isfinite(results['image_price']['price_sar'])
    assert results['identity']['الاقتراح_الرئيسي']['الماركة']
    print('All three real API predictions passed.', flush=True)

if __name__ == '__main__':
    main()
