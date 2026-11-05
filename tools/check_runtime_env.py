"""تحقق من عزل بيئة Python واعتمادياتها دون PYTHONPATH أو مكتبات النظام."""
import argparse
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import site
import sys

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--image", action="store_true")
args = parser.parse_args()
assert not os.environ.get("PYTHONPATH"), "أزل PYTHONPATH قبل التحقق"
assert sys.prefix != sys.base_prefix, "استخدم بيئة venv مستقلة"
assert site.ENABLE_USER_SITE is False
root = Path(sys.prefix).resolve()
packages = {"numpy": "numpy", "scipy": "scipy", "sklearn": "scikit-learn", "torch": "torch", "torchvision": "torchvision", "transformers": "transformers", "PIL": "Pillow", "pandas": "pandas"}
if not args.image:
    packages.update({"gradio": "gradio", "fastapi": "fastapi", "sqlalchemy": "SQLAlchemy"})
versions = {}
for module_name, distribution in packages.items():
    module = importlib.import_module(module_name)
    assert Path(module.__file__).resolve().is_relative_to(root), module_name
    versions[distribution] = importlib.metadata.version(distribution)
assert versions["scikit-learn"] == ("1.6.1" if args.image else "1.9.1")
print(json.dumps({"isolated": True, "python": sys.version.split()[0], "user_site": False, "pythonpath": False, "versions": versions}, ensure_ascii=False))
