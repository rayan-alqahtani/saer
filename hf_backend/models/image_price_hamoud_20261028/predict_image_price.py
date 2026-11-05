"""استدلال نموذج حمود للسعر من صورة السيارة فقط."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import image_to_price_v2 as model_code
import joblib


def predict(paths: list[Path]) -> dict:
    if not 1 <= len(paths) <= 8:
        raise ValueError("ارفع من صورة واحدة إلى 8 صور للسيارة نفسها.")
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)

    bundle = joblib.load(HERE / "bundle.joblib")
    needed = sorted({name for comp in bundle["components"] for name in comp["backbones"]})
    embeddings = {}
    for name in needed:
        # حد الدفعة يجعل استدلال DINOv2-Large مناسبًا لذاكرة الجهاز المحلي.
        batch_size = 2 if name == "dinov2_l" else 4
        e, ef = model_code._run_encoder(
            name, [str(path) for path in paths], batch=batch_size, workers=0
        )
        embeddings[name] = (model_code.l2norm(e), model_code.l2norm(ef))

    log_price = float(np.mean([
        model_code.comp_logpred(comp, embeddings)
        for comp in bundle["components"]
    ]))
    q = bundle["q"]
    return {
        "status": "experimental_estimate",
        "images_used": len(paths),
        "price_sar": float(np.exp(log_price)),
        "range_80_sar": [float(np.exp(log_price + q["lo80"])),
                         float(np.exp(log_price + q["hi80"]))],
        "range_90_sar": [float(np.exp(log_price + q["lo90"])),
                         float(np.exp(log_price + q["hi90"]))],
        "notice_ar": "تقدير إعلاني تجريبي من الصور فقط؛ لا يوضح الممشى أو الحالة، وليس سعر بيع مضمونًا.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="تقدير تجريبي لسعر السيارة من الصورة")
    parser.add_argument("images", nargs="+", type=Path, help="مسار صورة واحدة إلى 8 صور للسيارة نفسها")
    args = parser.parse_args()
    torch.set_num_threads(8)
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8")
        print(json.dumps(predict(args.images), ensure_ascii=False, indent=2))
    except Exception as exc:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8")
        print(json.dumps({"status": "error", "message_ar": str(exc)}, ensure_ascii=False, indent=2))
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
