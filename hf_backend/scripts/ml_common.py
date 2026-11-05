"""هوية تشغيل النماذج وحفظ أدلة التدريب دون تغيير البيانات."""
import os
for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_name, '4')
import sys
import platform
import importlib.metadata
from datetime import datetime, timezone
from day20_common import ROOT, PROCESSED, read_csv, load_json, save_json, save_csv, save_bytes, sha256, now
from day21_dataset_contract import dataset_manifest, assert_artifact_compatible

SEED = 42


def verified_inputs():
    gate = load_json(ROOT / 'data/audit/day20/data_completion_gate.json')
    if not gate.get('training_allowed'):
        raise ValueError('بوابة التدريب لا تسمح بالتشغيل')
    expected = load_json(PROCESSED / 'training_manifest.json')
    actual = dataset_manifest(PROCESSED, expected['dataset_version'])
    assert_artifact_compatible(actual, expected)
    if actual['input_files_sha256'] != expected['input_files_sha256']:
        raise ValueError('تغير أحد ملفات البيانات المعتمدة')
    publication = load_json(ROOT / 'data/audit/day20/content_publication.json')
    for name, digest in publication['processed_files_sha256'].items():
        if sha256(PROCESSED / name) != digest:
            raise ValueError('تغير الملف بعد التحقق النهائي: ' + name)
    return {k: expected[k] for k in expected['required_model_metadata']}


def new_run(task):
    identity = verified_inputs()
    folder = ROOT / 'models' / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f') + '_' + task)
    folder.mkdir(parents=True, exist_ok=False)
    env = {}
    for name in ('numpy', 'scipy', 'scikit-learn', 'joblib', 'Pillow', 'torch', 'torchvision'):
        try: env[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: env[name] = None
    metadata = {**identity, 'task': task, 'created_at': now(), 'seed': SEED,
                'python': sys.version, 'python_executable': sys.executable,
                'platform': platform.platform(), 'device': 'cpu', 'versions': env,
                'training_status': 'started', 'external_evaluation': 'لم يختبر بعد',
                'source_review': {str(p.relative_to(ROOT)): sha256(p) for p in [
                    ROOT/'external_group_c/02_price_model.py', ROOT/'external_group_c/03_train_recognizer.py',
                    ROOT/'external_group_c/04_visual_search.py']}}
    save_json(folder / 'metadata.json', metadata)
    gate_path = ROOT / 'data/audit/day20/data_completion_gate.json'
    gate = load_json(gate_path)
    # إعلان بدء التدريب يكتب مرة واحدة؛ بقية المهام لها metadata مستقلة.
    # تجنب تنافس مهمتين على الملف المرحلي المشترك لسجل البوابة.
    if not gate.get('training_started'):
        gate.update(training_started=True, training_allowed=True, updated_at=now(), status='training_authorized_in_progress')
        save_json(gate_path, gate)
    return folder, metadata


def assert_split_integrity(rows, image_task=False):
    seen_groups, seen_hashes = {}, {}
    for r in rows:
        if r['split'] not in ('train', 'val', 'test'): raise ValueError('قسم غير صالح')
        group = r['group_id']
        if group in seen_groups and seen_groups[group] != r['split']: raise ValueError('تسرب مجموعة')
        seen_groups[group] = r['split']
        if image_task:
            digest = r['sha256']
            if digest in seen_hashes: raise ValueError('صورة مكررة في مهمة التدريب')
            seen_hashes[digest] = r['split']
    if set(seen_groups.values()) != {'train', 'val', 'test'}: raise ValueError('قسم فارغ')


def log(folder, message):
    print(message, flush=True)
    with (folder / 'events.log').open('a', encoding='utf-8') as f:
        f.write(now() + ' ' + message + '\n')
