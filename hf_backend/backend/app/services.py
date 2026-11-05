from __future__ import annotations

import hashlib
import os
import subprocess
import importlib.metadata
import json
import sys
import threading
from pathlib import Path
from typing import Any

from backend.app.config import (
    IDENTITY_CACHE_DIR,
    IMAGE_PRICE_CACHE_DIR,
    ROOT,
)

_lock = threading.Lock()
_tabular_runtime: Any = None
_identity_runtime: Any = None


def _cached(directory: Path, digest: str) -> dict | None:
    path = directory / f"{digest}.json"
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _save_cache(directory: Path, digest: str, result: dict) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{digest}.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def tabular_runtime():
    global _tabular_runtime
    if _tabular_runtime is None:
        scripts = str(ROOT / "scripts")
        if scripts not in sys.path:
            sys.path.insert(0, scripts)
        from backend.app.hf_runtime import HostedTabularRuntime

        _tabular_runtime = HostedTabularRuntime()
    return _tabular_runtime


def identity_runtime():
    global _identity_runtime
    if _identity_runtime is None:
        scripts = str(ROOT / "scripts")
        if scripts not in sys.path:
            sys.path.insert(0, scripts)
        from backend.app.hf_runtime import HostedIdentityRuntime

        _identity_runtime = HostedIdentityRuntime()
    return _identity_runtime


def supported_price_pairs() -> dict[str, list[str]]:
    runtime = tabular_runtime()
    counts = runtime.price_model.features.model_counts
    pairs: dict[str, list[str]] = {}
    for item in counts:
        if "|" not in item:
            continue
        make, model = item.split("|", 1)
        pairs.setdefault(str(make), []).append(str(model))
    return {make: sorted(set(models)) for make, models in sorted(pairs.items())}


def predict_tabular(payload: dict) -> dict:
    with _lock:
        return tabular_runtime().predict("price", row=payload)


def predict_identity(path: Path, digest: str) -> dict:
    cached = _cached(IDENTITY_CACHE_DIR, digest)
    if cached is not None:
        return {**cached, "من_ذاكرة_مؤقتة": True}
    with _lock:
        result = identity_runtime().predict("recognize", path)
    _save_cache(IDENTITY_CACHE_DIR, digest, result)
    return {**result, "من_ذاكرة_مؤقتة": False}


def predict_image_price(paths: Path | list[Path], digest: str) -> dict:
    cached = _cached(IMAGE_PRICE_CACHE_DIR, digest)
    if cached is not None:
        return {**cached, "من_ذاكرة_مؤقتة": True}
    image_paths = [paths] if isinstance(paths, Path) else paths
    if os.environ.get("SAER_IMAGE_PYTHON"):
        env = os.environ.copy()
        if os.environ.get("SAER_IMAGE_PYTHONPATH"):
            env["PYTHONPATH"] = os.environ["SAER_IMAGE_PYTHONPATH"]
        else:
            env.pop("PYTHONPATH", None)
        worker = ROOT / "scripts" / "saer_price_worker.py"
        with _lock:
            completed = subprocess.run([os.environ["SAER_IMAGE_PYTHON"], str(worker), *map(str, image_paths)],
                cwd=ROOT, env=env, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=900)
        if completed.returncode:
            raise RuntimeError("تعذر تشغيل عامل نموذج الصورة: " + completed.stderr[-500:])
        result = json.loads(completed.stdout.strip().splitlines()[-1])
        _save_cache(IMAGE_PRICE_CACHE_DIR, digest, result)
        return {**result, "من_ذاكرة_مؤقتة": False}
    if importlib.metadata.version("scikit-learn") != "1.6.1":
        raise RuntimeError("نموذج الصورة يتطلب بيئة scikit-learn 1.6.1 منفصلة؛ اضبط SAER_IMAGE_PYTHON")
    try:
        model_path = ROOT / "models" / "image_price_hamoud_20261028"
        if str(model_path) not in sys.path:
            sys.path.insert(0, str(model_path))
        from predict_image_price import predict

        with _lock:
            result = predict(image_paths)
    except (ValueError, OSError, RuntimeError, TimeoutError) as exc:
        raise RuntimeError(f"تعذر تشغيل نموذج حمود: {str(exc)[:800]}") from exc
    _save_cache(IMAGE_PRICE_CACHE_DIR, digest, result)
    return {**result, "من_ذاكرة_مؤقتة": False}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
