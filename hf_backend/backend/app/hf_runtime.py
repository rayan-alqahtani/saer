from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import joblib

from backend.app.config import ROOT


class HostedTabularRuntime:
    """حمولة تشغيل صغيرة لنموذج السعر الجدولي المعتمد فقط."""

    def __init__(self) -> None:
        scripts = str(ROOT / "scripts")
        if scripts not in sys.path:
            sys.path.insert(0, scripts)
        from car_model_runtime import validate_price

        pointer = json.loads((ROOT / "models" / "improvement_current.json").read_text(encoding="utf-8"))
        run_dir = str(pointer["run_dir"]).replace("\\", "/")
        model_path = ROOT / run_dir / "price" / "best_model.joblib"
        if not model_path.is_file():
            raise FileNotFoundError("ملف نموذج السعر الجدولي غير موجود في مستودع الأصول الخاص")
        self.price_model = joblib.load(model_path)
        self._validate_price = validate_price

    def predict(self, task: str, row: dict[str, Any]) -> dict[str, Any]:
        if task != "price":
            raise ValueError("النموذج الجدولي يستقبل مهمة السعر فقط")
        clean = self._validate_price(row, self.price_model.features.model_counts)
        value = float(self.price_model.predict([clean])[0])
        if not math.isfinite(value) or value <= 0:
            raise ValueError("تعذر إنتاج سعر صالح")
        return {
            "سعر_الإعلان_المقدر_ريال": round(value, 2),
            "الحالة": "تجريبي",
            "تنبيه": "سعر إعلان تجريبي للمستعمل اعتمادًا على المواصفات وبيانات التدريب؛ ليس سعر بيع مضمونًا.",
        }


class HostedIdentityRuntime:
    """يعيد استخدام مصنف الهوية المعتمد وشفرة الميزات الأصلية، دون تحميل تجارب قديمة."""

    def __init__(self) -> None:
        scripts = str(ROOT / "scripts")
        if scripts not in sys.path:
            sys.path.insert(0, scripts)
        from model_store import ensure_public_identity_backbones

        ensure_public_identity_backbones()
        import clip_identity_round_20261027 as clip

        pointer = json.loads((ROOT / "models" / "identity_current.json").read_text(encoding="utf-8"))
        if pointer.get("status") != "final_approved_for_identity_mvp":
            raise ValueError("سجل اعتماد نموذج الهوية غير صالح")
        entry = pointer.get("tasks", {}).get("model", {})
        artifact = str(entry.get("artifact", "")).replace("\\", "/")
        path = ROOT / artifact
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != entry.get("sha256"):
            raise ValueError("وزن نموذج الهوية غير موجود أو بصمته غير مطابقة")

        # تجاوز مُهيّئ واجهة التجارب الذي يتحقق من مجموعة تجارب التدريب كاملة؛
        # التنبؤ نفسه يستعمل extract/clip_probabilities/predict من الشفرة الأصلية.
        self.selection: dict[str, Any] = {}
        self.foreground_models = None
        self.geometry_models = None
        self.identity_head = joblib.load(path)
        self.clip_model, self.clip_processor = clip.load_clip()

    @staticmethod
    def _top(probabilities, classes, limit: int = 3) -> list[dict[str, Any]]:
        import numpy as np

        values = np.asarray(probabilities).reshape(-1)
        order = np.argsort(-values)[:limit]
        return [
            {"الفئة": str(classes[index]), "درجة_غير_معايرة": float(values[index])}
            for index in order
        ]

    def extract(self, path: Path) -> dict[str, Any]:
        """استخدم مستخرجَي DINO وCLIP الأصليين مع إبقاء الأوزان على الجهاز الحالي."""
        import sys

        scripts = str(ROOT / "scripts")
        if scripts not in sys.path:
            sys.path.insert(0, scripts)
        import visual_fusion_predict as visual
        import clip_identity_round_20261027 as clip
        import torch
        from PIL import Image
        from image_input_policy_20261026 import decode_rgb

        features = visual.extract([path])[0]
        with Image.open(path) as source:
            image = decode_rgb(source)
        crop = image.crop(features["box"]) if features["detected"] else image
        with torch.inference_mode():
            inputs = self.clip_processor(images=[image, crop], return_tensors="pt")
            device = next(self.clip_model.parameters()).device
            inputs = {key: value.to(device) for key, value in inputs.items()}
            features["_clip"] = clip.norm(
                self.clip_model.get_image_features(**inputs).float().cpu().numpy()
            )
        return features

    def clip_probabilities(self, task: str, features: dict[str, Any]):
        if task != "model":
            raise ValueError("نموذج الهوية يستقبل مهمة الموديل المركب فقط")
        import numpy as np
        import clip_identity_round_20261027 as clip

        blocks = [
            clip.norm(np.asarray(features["full"])[None]),
            clip.norm(np.asarray(features["crop"])[None]),
            clip.norm(np.asarray(features["_clip"])[0][None]),
            clip.norm(np.asarray(features["_clip"])[1][None]),
        ]
        x = np.concatenate(blocks, axis=1)
        head = self.identity_head
        return head["model"].predict_proba(head["scaler"].transform(x)), head["classes"]

    def predict(self, task: str, path: Path) -> dict[str, Any]:
        if task != "recognize":
            raise ValueError("نموذج الهوية يدعم التعرف على السيارة فقط")
        features = self.extract(path)
        probabilities, classes = self.clip_probabilities("model", features)
        top = self._top(probabilities[0], classes)
        label = top[0]["الفئة"] if top else "غير متوفر"
        make, _, model = label.partition("|")
        return {
            "الحالة": "معتمد نهائيًا لمسار التعرف على الهوية؛ الدليل الخارجي محدود",
            "الموديل_المستقل": {"النتيجة": label, "أفضل_الاحتمالات": top},
            "مصدر_الماركة": "مستخرجة من فئة الموديل المركبة",
            "الاقتراح_الرئيسي": {
                "النتيجة": label,
                "الماركة": make or "غير محدد",
                "الموديل": model or label,
                "درجة_غير_معايرة": top[0]["درجة_غير_معايرة"] if top else 0.0,
                "أفضل_الاحتمالات": top,
            },
            "المسار": "DINOv2 + CLIP؛ رأس مصنف الماركة والموديل المعتمد",
            "نطاق_الموديل": "نتيجة تجريبية؛ test الداخلي لا يثبت التعميم الخارجي.",
            "يلزم_تأكيد_المستخدم_قبل_التسعير": True,
        }
