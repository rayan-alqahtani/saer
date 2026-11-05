"""Approved DINOv2+CLIP identity heads for the image tester only."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np

import clip_identity_round_20261027 as clip
from clip_identity_runtime_20261028 import ClipIdentityRuntime
from model_image_tester_20261026 import CandidateRuntime

ROOT = Path(__file__).resolve().parents[1]
POINTER = ROOT / "models" / "identity_current.json"
ARTIFACTS = ROOT / "outputs" / "generalization_rescue_20261028_v1"


class ApprovedIdentityRuntime(ClipIdentityRuntime):
    def __init__(self):
        # The approved joint head postdates CandidateRuntime's old baseline
        # pointer. Initialize the frozen feature pipeline directly instead of
        # requiring that obsolete pointer; retain artifact/code hash checks.
        import foreground_visual_round_20261026 as fg

        manifest = fg.r.v.load_json(ROOT / "outputs" / "visual_rescue_20261026" / "runtime_code_manifest.json")
        for name, expected in manifest["files"].items():
            if fg.r.digest(ROOT / name) != expected:
                raise ValueError("تغير ملف تشغيل مسار الهوية: " + name)
        self.selection = fg.r.v.load_json(ROOT / "outputs" / "visual_rescue_20261026" / "chosen_development_candidates.json")["tasks"]
        self.artifacts = {}
        self.foreground_models = None
        self.geometry_models = None
        for task, item in self.selection.items():
            artifact = ROOT / item["artifact"]
            if fg.r.digest(artifact) != item["sha256"]:
                raise ValueError("تغير وزن المرشح: " + task)
            self.artifacts[task] = joblib.load(artifact)

        self.clip_decisions = clip.r.v.load_json(clip.OUT / "results.json")
        self.clip_heads = {
            task: joblib.load(clip.OUT / task / "head.joblib")
            for task, decision in self.clip_decisions.items()
            if decision["accepted"]
        }
        if not self.clip_heads:
            raise ValueError("لا يوجد رأس CLIP اجتاز بوابة التطوير المحفوظة")
        release = clip.r.v.load_json(clip.OUT / "runtime_release.json")
        for name, expected in release["files"].items():
            if clip.r.digest(clip.ROOT / name) != expected:
                raise ValueError("تغير ملف تشغيل CLIP: " + name)
        self.clip_model, self.clip_processor = clip.load_clip()
        self.clip_text = {task: np.load(clip.OUT / f"{task}_text.npy") for task in self.clip_heads}
        self.clip_scale = clip.r.v.load_json(clip.OUT / "extraction_complete.json")["text_logit_scale"]
        self.root = clip.OUT

        self.approved_identity = json.loads(POINTER.read_text(encoding="utf-8"))
        if self.approved_identity.get("status") != "final_approved_for_identity_mvp":
            raise ValueError("سجل الاعتماد النهائي لنموذج الهوية غير صالح")
        # Use the single joint Make|Model head so its brand/model pair cannot
        # conflict with an independent make classifier in the user interface.
        task = "model"
        entry = self.approved_identity["tasks"][task]
        path = ROOT / entry["artifact"]
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != entry["sha256"]:
            raise ValueError("بصمة رأس الموديل لا تطابق سجل الاعتماد")
        self.identity_head = joblib.load(path)

    def clip_probabilities(self, task, features):
        if task != "model":
            return super().clip_probabilities(task, features)
        head = self.identity_head
        blocks = [
            clip.norm(np.asarray(features["full"])[None]),
            clip.norm(np.asarray(features["crop"])[None]),
            clip.norm(np.asarray(features["_clip"])[0][None]),
            clip.norm(np.asarray(features["_clip"])[1][None]),
        ]
        x = np.concatenate(blocks, axis=1)
        return head["model"].predict_proba(head["scaler"].transform(x)), head["classes"]

    def predict(self, task, path, k=5, features=None):
        if task != "recognize":
            return super().predict(task, path, k, features)
        f = features if features is not None else self.extract(path)
        model_p, model_classes = self.clip_probabilities("model", f)
        model_top = self._top(model_p[0], model_classes)
        model_name = model_top[0]["الفئة"] if model_top else "غير متوفر"
        model_make, _, short_model = model_name.partition("|")
        return {
            "الحالة": "معتمد نهائيًا لمسار التعرف على الهوية؛ الدليل الخارجي محدود",
            "الموديل_المستقل": {"النتيجة": model_name, "أفضل_الاحتمالات": model_top},
            "مصدر_الماركة": "مستخرجة من فئة الموديل المركبة",
            "الاقتراح_الرئيسي": {
                "النتيجة": model_name,
                "الماركة": model_make or "غير محدد",
                "الموديل": short_model or model_name,
                "درجة_غير_معايرة": model_top[0]["درجة_غير_معايرة"] if model_top else 0.0,
                "أفضل_الاحتمالات": model_top,
            },
            "المسار": "DINOv2 + CLIP؛ رأس جديد للماركة والموديل فقط",
            "نطاق_الموديل": "نتيجة تجريبية؛ test داخلي سبق استخدامه تاريخيًا ولا يثبت التعميم الخارجي.",
            "يلزم_تأكيد_المستخدم_قبل_التسعير": True,
        }


if __name__ == "__main__":
    import tkinter as tk
    from model_image_tester import ModelImageTester

    root = tk.Tk()
    root.title("سعّر — تجربة نماذج الهوية المعتمدة")
    ModelImageTester(
        root,
        engine_factory=ApprovedIdentityRuntime,
        subtitle="الماركة والموديل: DINOv2 + CLIP. السعر لا يُستنتج من هذا المصنف.",
    )
    root.mainloop()
