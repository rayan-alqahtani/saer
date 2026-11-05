from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any

from backend.app.config import (
    IDENTITY_CACHE_DIR,
    IMAGE_PRICE_CACHE_DIR,
    IMAGE_PRICE_RUNTIME,
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
        from backend.app.runtime_bridge import hosted_runtime

        _tabular_runtime = hosted_runtime().HostedTabularRuntime()
    return _tabular_runtime


def identity_runtime():
    global _identity_runtime
    if _identity_runtime is None:
        scripts = str(ROOT / "scripts")
        if scripts not in sys.path:
            sys.path.insert(0, scripts)
        from backend.app.runtime_bridge import hosted_runtime

        _identity_runtime = hosted_runtime().HostedIdentityRuntime()
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


def _python_for_image_price() -> Path:
    configured = os.environ.get("SAER_IMAGE_PYTHON", "").strip()
    if configured:
        path = Path(configured)
        if not path.is_file():
            raise RuntimeError("مسار SAER_IMAGE_PYTHON غير صالح")
        return path
    candidates = [
        ROOT / "tmp" / "price_check_env" / "Scripts" / "python.exe",
        ROOT / ".venv" / "Scripts" / "python.exe",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise RuntimeError("بيئة تشغيل نموذج حمود غير موجودة")


def predict_image_price(paths: Path | list[Path], digest: str) -> dict:
    cached = _cached(IMAGE_PRICE_CACHE_DIR, digest)
    if cached is not None:
        return {**cached, "من_ذاكرة_مؤقتة": True}
    worker = ROOT / "scripts" / "saer_price_worker.py"
    image_paths = [paths] if isinstance(paths, Path) else paths
    command = [str(_python_for_image_price()), str(worker), *(str(path) for path in image_paths)]
    env = os.environ.copy()
    if os.environ.get("SAER_IMAGE_PYTHONPATH"):
        env["PYTHONPATH"] = os.environ["SAER_IMAGE_PYTHONPATH"]
    elif os.environ.get("SAER_IMAGE_PYTHON"):
        env.pop("PYTHONPATH", None)
    env.setdefault("OMP_NUM_THREADS", "4")
    env.setdefault("MKL_NUM_THREADS", "4")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    try:
        with _lock:
            completed = subprocess.run(
                command,
                cwd=ROOT,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=900,
                env=env,
                check=False,
            )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("استغرق تقدير الصورة وقتًا أطول من الحد؛ جرّب صورة أصغر") from exc
    if completed.returncode != 0:
        tail = (completed.stderr or completed.stdout)[-1200:]
        raise RuntimeError(f"تعذر تشغيل نموذج حمود: {tail.strip()}")
    try:
        result = json.loads(completed.stdout.strip().splitlines()[-1])
    except (json.JSONDecodeError, IndexError) as exc:
        raise RuntimeError("لم يُرجع نموذج حمود نتيجة قابلة للقراءة") from exc
    if result.get("status") == "error":
        raise RuntimeError(result.get("message_ar") or "تعذر تقدير الصورة")
    _save_cache(IMAGE_PRICE_CACHE_DIR, digest, result)
    return {**result, "من_ذاكرة_مؤقتة": False}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
