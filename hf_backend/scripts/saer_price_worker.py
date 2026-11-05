"""عامل محلي يستدعي نموذج حمود المعتمد دون خلطه بمسار السعر الجدولي."""
from __future__ import annotations

import importlib.metadata
if importlib.metadata.version("scikit-learn") != "1.6.1":
    raise RuntimeError("نموذج الصورة يتطلب scikit-learn==1.6.1؛ حدد بيئة SAER_IMAGE_PYTHON المناسبة")

import contextlib
import io
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "models" / "image_price_hamoud_20261028"))
from predict_image_price import predict  # noqa: E402


def main() -> int:
    if len(sys.argv) < 2:
        print(json.dumps({"status": "error", "message_ar": "ارفع صورة واحدة على الأقل"}, ensure_ascii=False))
        return 2
    with contextlib.redirect_stdout(io.StringIO()):
        result = predict([Path(value) for value in sys.argv[1:]])
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
