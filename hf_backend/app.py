from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

# يجب تحميل تكامل ZeroGPU قبل استيراد أي مكتبة قد تحمل PyTorch.
try:
    import spaces
    gpu = spaces.GPU
except ImportError:
    def gpu(*, duration: int = 60):
        def decorate(function):
            return function
        return decorate

import gradio as gr
import uvicorn
from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware

from model_store import ROOT, ensure_private_assets, ensure_public_identity_clip_cache

os.environ.setdefault("SAER_APP_ROOT", str(ROOT))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

_startup_error: str | None = None
try:
    ensure_private_assets()
    ensure_public_identity_clip_cache()
except Exception as exc:
    # تبقى واجهة التشخيص عاملة مع رسالة صريحة بدلاً من إظهار نتائج ثابتة.
    _startup_error = f"تعذر تجهيز أصول النماذج: {type(exc).__name__}: {str(exc)[:500]}"

from backend.app import services

# ZeroGPU exposes CUDA emulation during module initialization. Load the frozen
# public encoders here so decorated inference calls reuse CUDA-resident models.
if _startup_error is None:
    try:
        import sys
        import torch

        if torch.cuda.is_available():
            price_code_dir = ROOT / "models" / "image_price_hamoud_20261028"
            if str(price_code_dir) not in sys.path:
                sys.path.insert(0, str(price_code_dir))
            import image_to_price_v2

            image_to_price_v2.preload_cuda_encoders()
            services.identity_runtime()
    except Exception as exc:
        _startup_error = f"تعذر تحميل المشفّرات عند بدء التشغيل: {type(exc).__name__}: {str(exc)[:500]}"


def _valid_image_path(value: str | None) -> Path:
    if isinstance(value, (list, tuple)):
        raise ValueError("ارفع صورة واحدة فقط للسيارة")
    if not isinstance(value, (str, Path)) or not value:
        raise ValueError("اختر صورة السيارة أولًا")
    if _startup_error:
        raise RuntimeError(_startup_error)
    path = Path(value)
    if not path.is_file() or not 0 < path.stat().st_size <= 12 * 1024 * 1024:
        raise ValueError("الصورة مفقودة أو فارغة أو حجمها يتجاوز 12 ميجابايت")
    from PIL import Image, UnidentifiedImageError

    try:
        with Image.open(path) as image:
            if image.format not in {"JPEG", "PNG", "WEBP"}:
                raise ValueError("الصيغ المدعومة: JPG وPNG وWebP")
            if image.width * image.height > 40_000_000:
                raise ValueError("أبعاد الصورة كبيرة جداً")
            image.verify()
    except (OSError, UnidentifiedImageError, Image.DecompressionBombError) as exc:
        raise ValueError("ملف الصورة غير صالح") from exc
    return path


def identify_vehicle(upload: str | None) -> dict[str, Any]:
    path = _valid_image_path(upload)
    return services.predict_identity(path, services.file_sha256(path))


def estimate_image_price(upload: str | list[str] | None) -> dict[str, Any]:
    paths = upload if isinstance(upload, list) else [upload]
    if len(paths) != 1:
        raise ValueError("ارفع صورة واحدة فقط للسيارة")
    checked = [_valid_image_path(path) for path in paths]
    digest = hashlib.sha256("|".join(services.file_sha256(path) for path in checked).encode()).hexdigest()
    result = services.predict_image_price(checked, digest)
    notice = f"تقدير تجريبي من {len(checked)} صورة؛ لا يرى الممشى أو الحالة وليس سعر بيع مضموناً."
    return {**result, "image_count": len(checked), "notice_ar": notice, "notice": notice}


def estimate_specs(payload: dict[str, Any] | str | None) -> dict[str, Any]:
    if _startup_error:
        raise RuntimeError(_startup_error)
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise ValueError("بيانات المواصفات ليست JSON صالحاً") from exc
    if not isinstance(payload, dict):
        raise ValueError("أرسل مواصفات السيارة ككائن JSON")
    row = {
        "make": payload.get("make", ""),
        "model_std": payload.get("model_std", ""),
        "year": payload.get("year"),
        "km_mileage": payload.get("km_mileage"),
        "trim_std": payload.get("trim_std", ""),
        "city": payload.get("city", ""),
        "transmission": payload.get("transmission", ""),
        "type_fuel": payload.get("type_fuel", ""),
        "condition": payload.get("condition", "used"),
    }
    return services.predict_tabular(row)


predict_identity = gpu(duration=240)(identify_vehicle)
predict_image = gpu(duration=240)(estimate_image_price)

with gr.Blocks(title="سعّر — خدمة التقدير الحي") as demo:
    gr.Markdown("# سعّر — واجهة API للنماذج المعتمدة")
    gr.Markdown("تقديرات تجريبية؛ البيانات التاريخية لا تمثل سعر السوق الحالي.")
    with gr.Tab("التعرف على السيارة"):
        identity_file = gr.File(label="صورة السيارة", file_types=[".jpg", ".jpeg", ".png", ".webp"], type="filepath")
        identity_button = gr.Button("تعرّف على السيارة")
        identity_result = gr.JSON(label="النتيجة")
        identity_button.click(predict_identity, identity_file, identity_result, api_name="identify_vehicle")
    with gr.Tab("السعر من الصورة"):
        price_files = gr.File(label="صورة السيارة", file_types=[".jpg", ".jpeg", ".png", ".webp"], file_count="single", type="filepath")
        price_button = gr.Button("قدّر السعر")
        price_result = gr.JSON(label="النتيجة")
        price_button.click(predict_image, price_files, price_result, api_name="predict_image")
    with gr.Tab("السعر من المواصفات"):
        specs_input = gr.JSON(label="مواصفات السيارة بصيغة JSON")
        specs_button = gr.Button("قدّر السعر")
        specs_result = gr.JSON(label="النتيجة")
        specs_button.click(estimate_specs, specs_input, specs_result, api_name="predict_specs")


def create_app() -> FastAPI:
    origins = _allowed_origins()
    base = FastAPI()
    base.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )
    return gr.mount_gradio_app(base, demo, path="", max_file_size=15 * 1024 * 1024)


def _allowed_origins() -> list[str]:
    local_origins = [
        "http://localhost:5173", "http://127.0.0.1:5173",
        "http://localhost:4173", "http://127.0.0.1:4173",
        "http://localhost:4175", "http://127.0.0.1:4175",
    ]
    configured = [origin.strip().rstrip("/") for origin in os.environ.get("FRONTEND_ORIGINS", "").split(",") if origin.strip()]
    return sorted(set(local_origins + configured))


def launch_gradio_share(*, prevent_thread_lock: bool = False):
    """شغّل Gradio مع رابط HTTPS مؤقت للاستخدام المحلي أو Kaggle."""
    from starlette.middleware import Middleware

    result = demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", "7860")),
        share=True,
        prevent_thread_lock=prevent_thread_lock,
        show_error=True,
        max_file_size=15 * 1024 * 1024,
        app_kwargs={
            "middleware": [
                Middleware(
                    CORSMiddleware,
                    allow_origins=_allowed_origins(),
                    allow_credentials=False,
                    allow_methods=["GET", "POST", "OPTIONS"],
                    allow_headers=["*"],
                )
            ]
        },
    )
    share_url = next(
        (value for value in result if isinstance(value, str) and ".gradio.live" in value),
        None,
    ) if isinstance(result, (tuple, list)) else None
    if share_url:
        print(f"رابط API العام: {share_url}")
    else:
        share_url = getattr(demo, "share_url", None)
        if share_url:
            print(f"رابط API العام: {share_url}")
    return result


app = create_app()

if __name__ == "__main__":
    if os.environ.get("SAER_GRADIO_SHARE", "").strip().lower() in {"1", "true", "yes"}:
        launch_gradio_share()
    else:
        uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "7860")))
