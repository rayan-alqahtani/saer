from __future__ import annotations

import os
import shutil
from pathlib import Path

from huggingface_hub import snapshot_download

ROOT = Path(os.environ.get("SAER_APP_ROOT", Path(__file__).resolve().parent)).resolve()
PUBLIC_CLIP_CONFIG_REVISION = "3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268"
PUBLIC_CLIP_WEIGHTS_REVISION = "c237dc49a33fc61debc9276459120b7eac67e7ef"
PRIVATE_ASSET_PATTERNS = [
    "models/image_price_current.json",
    "models/image_price_hamoud_20261028/bundle.joblib",
    "models/improvement_current.json",
    "models/improvement_20261021T093545513846/price/best_model.joblib",
    "models/identity_current.json",
    "outputs/generalization_rescue_20261028_v1/model/candidate.joblib",
]


def _asset_roots() -> list[Path]:
    candidates: list[Path] = []
    configured = os.environ.get("LOCAL_ASSETS_DIR", "").strip()
    if configured:
        candidates.append(Path(configured).expanduser().resolve())

    candidates.append(ROOT)
    kaggle_input = Path("/kaggle/input")
    if kaggle_input.is_dir():
        candidates.extend(sorted(path for path in kaggle_input.iterdir() if path.is_dir()))

    unique: list[Path] = []
    seen: set[str] = set()
    for path in candidates:
        key = str(path)
        if key not in seen:
            seen.add(key)
            unique.append(path)
    return unique


def _has_private_assets(root: Path) -> bool:
    return all((root / relative).is_file() for relative in PRIVATE_ASSET_PATTERNS)


def _copy_private_assets(source: Path) -> None:
    for relative in PRIVATE_ASSET_PATTERNS:
        source_file = source / relative
        target_file = ROOT / relative
        target_file.parent.mkdir(parents=True, exist_ok=True)
        if source_file.resolve() != target_file.resolve():
            shutil.copy2(source_file, target_file)


def ensure_private_assets() -> None:
    for local_root in _asset_roots():
        if _has_private_assets(local_root):
            _copy_private_assets(local_root)
            print(f"استخدمت الأصول المحلية من: {local_root}")
            return

    repo_id = os.environ.get("HF_MODEL_REPO", "").strip()
    token = os.environ.get("HF_TOKEN")
    if not repo_id or not token:
        raise RuntimeError(
            "لم أجد الأصول الستة محلياً. ضعها تحت LOCAL_ASSETS_DIR أو أضف "
            "HF_MODEL_REPO وHF_TOKEN لتنزيلها من مستودع خاص."
        )
    snapshot_download(
        repo_id=repo_id,
        repo_type="model",
        token=token,
        local_dir=ROOT,
        allow_patterns=PRIVATE_ASSET_PATTERNS,
        max_workers=4,
    )


def ensure_public_identity_clip_cache() -> None:
    """يحمّل ملفات CLIP العامة إلى ذاكرة Hub قبل أن يطلب runtime القراءة محلياً."""
    cache_dir = Path(os.environ.get("HF_HUB_CACHE", Path.home() / ".cache" / "huggingface" / "hub"))
    common = {"repo_id": "openai/clip-vit-base-patch32", "repo_type": "model", "cache_dir": cache_dir}
    snapshot_download(revision=PUBLIC_CLIP_CONFIG_REVISION, allow_patterns=[
        "config.json", "preprocessor_config.json", "tokenizer.json", "tokenizer_config.json",
        "special_tokens_map.json", "vocab.json", "merges.txt", "model.safetensors.index.json",
    ], **common)
    snapshot_download(revision=PUBLIC_CLIP_WEIGHTS_REVISION, allow_patterns=["model.safetensors"], **common)


def ensure_public_identity_backbones() -> None:
    """حمّل كواشف الهوية العامة مرة واحدة عبر مستخرج المشروع المحفوظ في الذاكرة."""
    import sys
    import torch
    from torchvision.models import ResNet18_Weights, resnet18
    from torchvision.models.detection import (
        SSDLite320_MobileNet_V3_Large_Weights,
        ssdlite320_mobilenet_v3_large,
    )

    # أنشئ كاش المستودع العام وملفات checkpoint التي يقرأها مستخرج المشروع.
    torch.hub.load("facebookresearch/dinov2", "dinov2_vits14", pretrained=True, trust_repo=True)
    resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    ssdlite320_mobilenet_v3_large(weights=SSDLite320_MobileNet_V3_Large_Weights.DEFAULT)

    scripts = str(ROOT / "scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    import visual_fusion_predict

    visual_fusion_predict.ensure_models()

