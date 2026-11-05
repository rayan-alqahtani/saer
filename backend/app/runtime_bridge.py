"""تحميل runtime التسليم الصغير دون اشتراط تجارب التدريب المؤجلة."""
from functools import lru_cache
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys

@lru_cache(maxsize=1)
def hosted_runtime():
    root = Path(__file__).resolve().parents[2]
    directory = root / 'hf_backend'
    if str(directory) not in sys.path:
        sys.path.append(str(directory))
    spec = spec_from_file_location('delivery_hosted_runtime', directory / 'backend/app/hf_runtime.py')
    if spec is None or spec.loader is None:
        raise RuntimeError('ملف تشغيل النماذج المعتمدة غير متوفر')
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
