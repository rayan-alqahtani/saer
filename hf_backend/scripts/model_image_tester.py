"""واجهة محلية لاختبار مرشحي الصور التطويريين في سعّر."""
from __future__ import annotations

import json
import queue
import sys
import threading
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from PIL import Image, ImageOps, ImageTk
from image_input_policy_20261026 import decode_rgb, normalized_input_path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import visual_fusion_predict  # noqa: E402

IMAGE_TYPES = ("*.jpg *.jpeg *.png *.webp",)


class VisualFusionRuntime:
    """واجهة تشغيل للحزمة المرشحة؛ لا تغيّر مؤشر النماذج المعتمد في المشروع."""

    def __init__(self) -> None:
        train = visual_fusion_predict.train
        # The experiment lock also recorded the model pointer that was active
        # when it was built. The user may have since restored the same baseline
        # bundle under that pointer, so validate its target and all other locks.
        old = train.old
        comparison, tables = old.compare_inputs()
        if comparison["critical"] or comparison["differences"]:
            raise RuntimeError("بيانات الإدخال لا تطابق بروتوكول المرشح")
        if not all(item["authorized_match"] for item in comparison["files"].values()):
            raise RuntimeError("تعذر التحقق من بصمات ملفات الإدخال")
        for flag in ("usable_make_recognition", "usable_recognition", "usable_color", "usable_body_type", "usable_visual_search"):
            split_by_sha: dict[str, set[str]] = {}
            for row in tables["images.csv"]:
                if row.get(flag) == "True":
                    split_by_sha.setdefault(row["sha256"], set()).add(row["split"])
            if any(len(splits) > 1 for splits in split_by_sha.values()):
                raise RuntimeError(f"تداخل بصمات بين الأقسام في {flag}")

        pointer_path = train.ROOT / "models" / "improvement_current.json"
        expected_run = str(old.OLD.relative_to(train.ROOT)).replace("/", "\\")
        pointer = train.load_json(pointer_path)
        if str(pointer.get("run_dir", "")).replace("/", "\\") != expected_run:
            raise RuntimeError("مؤشر النموذج المرجعي لا يطابق المرجع المثبت في المرشح")

        for protocol_path in (train.RUN / "protocol.json", old.RUN / "protocol.json"):
            protocol = train.load_json(protocol_path)
            for relative, expected in protocol["protected"].items():
                target = Path(relative)
                if not target.is_absolute():
                    target = train.ROOT / target
                if target.resolve() == pointer_path.resolve():
                    continue  # The referenced baseline was checked semantically above.
                if train.sha256(target) != expected:
                    raise RuntimeError(f"فشل قفل سلامة الملف: {target}")

        lock = train.load_json(train.RUN / "final_lock.json")
        for relative, expected in lock["artifacts"].items():
            if train.sha256(train.RUN / relative) != expected:
                raise RuntimeError(f"ملف مرشح الصور تغيّر: {relative}")
        for relative, expected in lock["sources"].items():
            if train.sha256(train.ROOT / relative) != expected:
                raise RuntimeError(f"كود المرشح تغيّر: {relative}")
        self.root = train.RUN

    def extract(self, path: Path) -> dict:
        return visual_fusion_predict.extract([path])[0]

    @staticmethod
    def _top(probabilities, classes, limit: int = 3) -> list[dict]:
        import numpy as np

        values = np.asarray(probabilities).reshape(-1)
        order = np.argsort(-values)[:limit]
        return [
            {"الفئة": str(classes[index]), "درجة_غير_معايرة": float(values[index])}
            for index in order
        ]

    @staticmethod
    def _arabic_label(task: str, label: str) -> str:
        translations = {
            "color": {
                "beige": "بيج", "black": "أسود", "blue": "أزرق", "brown": "بني",
                "gold": "ذهبي", "gray": "رمادي", "green": "أخضر", "red": "أحمر",
                "silver": "فضي", "white": "أبيض",
            },
            "body": {
                "hatchback": "هاتشباك", "pickup": "بيك أب", "sedan": "سيدان",
                "suv_crossover": "SUV / كروس أوفر", "van": "فان",
            },
        }
        return translations.get(task, {}).get(label, label)

    def predict(self, task: str, path: Path, k: int = 5, features: dict | None = None) -> dict:
        if task == "recognize":
            make_p, make_classes = visual_fusion_predict.predict("make", path, features)
            model_p, model_classes = visual_fusion_predict.predict("model", path, features)
            make_top = self._top(make_p[0], make_classes)
            model_top = self._top(model_p[0], model_classes)
            make_name = make_top[0]["الفئة"] if make_top else "غير متوفر"
            model_name = model_top[0]["الفئة"] if model_top else "غير متوفر"
            model_make, separator, model_short_name = model_name.partition("|")
            if not separator:
                model_make, model_short_name = "غير محدد", model_name
            return {
                "الحالة": "مرشح تطويري",
                "الماركة_المستقلة": {"النتيجة": make_name, "أفضل_الاحتمالات": make_top},
                "الموديل_المستقل": {"النتيجة": model_top[0]["الفئة"] if model_top else "غير متوفر", "أفضل_الاحتمالات": model_top},
                "الاقتراح_الرئيسي": {
                    "النتيجة": model_name,
                    "الماركة": model_make,
                    "الموديل": model_short_name,
                    "درجة_غير_معايرة": model_top[0]["درجة_غير_معايرة"] if model_top else 0.0,
                    "أفضل_الاحتمالات": model_top,
                },
                "اتساق_الماركة_والموديل": model_make == make_name,
                "المسار": "مرشح 25 أكتوبر؛ درجة غير معايرة",
                "نطاق_الموديل": "يغطي فئات الموديل المدعومة فقط؛ الاقتراح غير مؤكد والدرجات غير معايرة",
            }
        if task in ("color", "body"):
            probabilities, classes = visual_fusion_predict.predict(task, path, features)
            result = self._top(probabilities[0], classes)
            for item in result:
                item["الفئة"] = self._arabic_label(task, item["الفئة"])
            selected = visual_fusion_predict.train.load_json(
                visual_fusion_predict.train.RUN / task / "result.json"
            )
            uses_candidate = bool(selected.get("accepted"))
            return {
                "النتيجة": result[0]["الفئة"] if result else "غير متوفر",
                "أفضل_النتائج": result,
                "الحالة": "مرشح تطويري" if uses_candidate else "المرجع المحفوظ",
                "درجة_القرار": result[0]["درجة_غير_معايرة"] if result else 0.0,
                "تنبيه": "الدرجات غير معايرة؛ نتيجة الصورة لا تثبت صحة السعر أو الحالة الميكانيكية.",
            }
        if task == "search":
            matches = visual_fusion_predict.search(path, features)[:k]
            return {
                "النتائج": matches,
                "تنبيه": "بحث بصري تجريبي؛ الإعلانات تاريخية ودرجة التشابه ليست ثقة معايرة.",
            }
        raise ValueError(f"مهمة غير مدعومة: {task}")


class ModelImageTester:
    def __init__(self, root: tk.Tk, engine_factory=VisualFusionRuntime, subtitle: str | None = None) -> None:
        self.root = root
        self.engine_factory = engine_factory
        root.title("سعّر — تجربة نماذج الصور")
        root.geometry("920x700")
        root.minsize(760, 600)
        root.configure(bg="#f4f7fb")

        self.engine = None
        self.image_path: Path | None = None
        self.preview_image = None
        self.events: queue.Queue = queue.Queue()
        self.busy = False

        style = ttk.Style(root)
        style.configure("TButton", font=("Segoe UI", 10), padding=7)
        style.configure("Header.TLabel", font=("Segoe UI", 20, "bold"), background="#f4f7fb", foreground="#14233b")
        style.configure("Sub.TLabel", font=("Segoe UI", 11), background="#f4f7fb", foreground="#526174")

        ttk.Label(root, text="تجربة نماذج الصور", style="Header.TLabel", anchor="e").pack(fill="x", padx=24, pady=(18, 2))
        ttk.Label(root, text=subtitle or "مرشح الصور التطويري — جولة 25 أكتوبر؛ أرقام التحقق استكشافية وليست اختبارًا خارجيًا.", style="Sub.TLabel", anchor="e").pack(fill="x", padx=24, pady=(0, 12))

        top = tk.Frame(root, bg="#f4f7fb")
        top.pack(fill="x", padx=24)
        self.preview = tk.Label(top, text="لم تُختر صورة بعد", width=36, height=13, bg="white", fg="#68778c", relief="solid", bd=1)
        self.preview.pack(side="right", padx=(14, 0))
        panel = tk.Frame(top, bg="#f4f7fb")
        panel.pack(side="right", fill="both", expand=True)
        self.file_label = tk.Label(panel, text="اختر صورة سيارة من جهازك", bg="#f4f7fb", fg="#334155", anchor="e", justify="right", wraplength=440, font=("Segoe UI", 10))
        self.file_label.pack(fill="x", pady=(4, 10))
        self.pick_button = ttk.Button(panel, text="اختيار صورة", command=self.choose_image)
        self.pick_button.pack(anchor="e", pady=(0, 12))

        buttons = tk.Frame(panel, bg="#f4f7fb")
        buttons.pack(fill="x", anchor="e")
        self.task_buttons = {}
        actions = [
            ("recognize", "التعرف على الماركة والموديل"),
            ("color", "تقدير اللون"),
            ("body", "تصنيف الهيكل"),
            ("search", "البحث عن صور مشابهة"),
        ]
        for task, label in actions:
            button = ttk.Button(buttons, text=label, command=lambda selected=task: self.run_tasks([selected]), state="disabled")
            button.pack(fill="x", pady=3)
            self.task_buttons[task] = button
        self.all_button = ttk.Button(buttons, text="تشغيل كل مهام الصورة", command=lambda: self.run_tasks(["recognize", "color", "body", "search"]), state="disabled")
        self.all_button.pack(fill="x", pady=(8, 3))

        self.status = tk.Label(root, text="جارٍ تحميل النماذج المحلية…", bg="#e8eef7", fg="#31445f", anchor="e", padx=12, pady=8, font=("Segoe UI", 10))
        self.status.pack(fill="x", padx=24, pady=(14, 8))

        self.output = tk.Text(root, height=15, wrap="word", font=("Segoe UI", 10), bg="white", fg="#172033", relief="solid", bd=1, padx=12, pady=10)
        self.output.pack(fill="both", expand=True, padx=24, pady=(0, 10))
        self.output.tag_configure("rtl", justify="right", spacing1=2, spacing3=5)
        self.output.insert("1.0", "النتائج ستظهر هنا.\n\nهذه الواجهة تختبر مرشحي الماركة والموديل واللون والهيكل والبحث. السعر ليس من مهام هذه الحزمة؛ نموذج السعر في الجولة الأخيرة جدولي ويحتاج بيانات السيارة.")
        self.output.tag_add("rtl", "1.0", "end")
        self.output.configure(state="disabled")
        tk.Label(root, text="النتائج تجريبية ضمن نطاق البيانات؛ البحث يعرض إعلانات محفوظة، وتصنيف الهيكل لا يكشف الحوادث أو الضرر.", bg="#f4f7fb", fg="#667085", font=("Segoe UI", 9), wraplength=860, justify="right").pack(fill="x", padx=24, pady=(0, 12))

        threading.Thread(target=self.load_engine, daemon=True).start()
        self.root.after(100, self.poll_events)

    def load_engine(self) -> None:
        try:
            engine = self.engine_factory()
            self.events.put(("loaded", engine))
        except Exception as exc:  # show a useful local setup error in the window
            self.events.put(("load_error", str(exc)))

    def choose_image(self) -> None:
        filename = filedialog.askopenfilename(title="اختر صورة سيارة", filetypes=[("صور السيارات", "*.jpg *.jpeg *.png *.webp"), ("كل الملفات", "*.*")])
        if not filename:
            return
        path = Path(filename)
        try:
            with Image.open(path) as source:
                image = decode_rgb(source)
                image.thumbnail((280, 250))
                photo = ImageTk.PhotoImage(image)
            self.preview.configure(image=photo, text="")
            self.preview_image = photo
            self.image_path = path
            self.file_label.configure(text=path.name)
            self.output_set("اختر مهمة من الأزرار، أو شغّل جميع مهام الصورة.")
            self.set_controls(True)
        except Exception as exc:
            messagebox.showerror("تعذر فتح الصورة", str(exc))

    def run_tasks(self, tasks: list[str]) -> None:
        if self.busy or self.engine is None:
            return
        if self.image_path is None:
            messagebox.showinfo("اختر صورة", "اختر صورة سيارة أولاً.")
            return
        path = self.image_path
        self.busy = True
        self.set_controls(False)
        self.status.configure(text="جارٍ تنفيذ التنبؤ…")
        self.output_set("جارٍ تحليل الصورة. قد يستغرق البحث وقتًا أطول عند تحميل معرض الصور أول مرة.")
        threading.Thread(target=self.predict_worker, args=(tasks, path), daemon=True).start()

    def predict_worker(self, tasks: list[str], path: Path) -> None:
        try:
            with normalized_input_path(path) as model_path:
                features = self.engine.extract(model_path)
                results = [(task, self.engine.predict(task, path=model_path, k=5, features=features)) for task in tasks]
            self.events.put(("result", results))
        except Exception as exc:
            self.events.put(("predict_error", str(exc)))

    @staticmethod
    def format_results(results: list[tuple[str, dict]]) -> str:
        names = {"recognize": "التعرف على السيارة", "color": "اللون", "body": "الهيكل", "search": "السيارات المشابهة"}
        lines = []
        for task, result in results:
            lines.append(f"【{names[task]}】")
            if task == "recognize":
                independent_make = result.get("الماركة_المستقلة", {})
                primary = result.get("الاقتراح_الرئيسي", {})
                primary_make = primary.get("الماركة", "غير محدد")
                primary_model = primary.get("الموديل", "غير متوفر")
                lines.append(f"أقرب اقتراح: {primary_make} {primary_model}")
                lines.append(f"درجة الموديل الخام غير المعايرة: {primary.get('درجة_غير_معايرة', 0):.3f}")
                lines.append(f"توقع الماركة المستقل: {independent_make.get('النتيجة', 'غير متأكد')}")
                alternatives = primary.get("أفضل_الاحتمالات", [])
                if alternatives:
                    lines.append("أفضل 3 اقتراحات من مصنف الموديل (درجات غير معايرة):")
                    for index, item in enumerate(alternatives[:3], 1):
                        label = str(item.get("الفئة", "غير متوفر")).replace("|", " ")
                        lines.append(f"  {index}. {label} — {item.get('درجة_غير_معايرة', 0):.3f}")
                if not result.get("اتساق_الماركة_والموديل", False):
                    lines.append("تحذير: مصنف الماركة المستقل يختلف عن ماركة أقرب موديل؛ أبقينا اقتراح الموديل مع تنبيه.")
                else:
                    lines.append("تنبيه: توافق الماركة والموديل لا يثبت صحة التعرف؛ الاقتراح غير مؤكد.")
                lines.append(f"المسار: {result.get('المسار', 'غير متوفر')}")
                lines.append(result.get("نطاق_الموديل", ""))
            elif task in ("color", "body"):
                lines.append(f"النتيجة: {result.get('النتيجة', 'غير متوفر')}")
                lines.append(f"الحالة: {result.get('الحالة', 'غير متوفر')} — درجة القرار {result.get('درجة_القرار', 0):.3f}")
                lines.append(result.get("تنبيه", ""))
            else:
                matches = result.get("النتائج", [])
                if not matches:
                    lines.append("لم تظهر نتائج مشابهة.")
                for index, match in enumerate(matches, 1):
                    lines.append(f"{index}. الإعلان {match.get('معرف_الإعلان', '—')} — تشابه {match.get('التشابه', 0):.3f}")
                lines.append(result.get("تنبيه", ""))
            lines.append("")
        return "\n".join(lines).strip()

    def poll_events(self) -> None:
        try:
            while True:
                event, payload = self.events.get_nowait()
                if event == "loaded":
                    self.engine = payload
                    self.status.configure(text=f"النماذج جاهزة للتجربة — {payload.root.name}")
                    self.set_controls(self.image_path is not None)
                elif event == "load_error":
                    self.status.configure(text="تعذر تحميل النماذج")
                    self.output_set(payload)
                    messagebox.showerror("تعذر تحميل الحزمة", payload)
                elif event == "result":
                    self.status.configure(text="اكتمل التنبؤ")
                    self.output_set(self.format_results(payload))
                    self.busy = False
                    self.set_controls(True)
                elif event == "predict_error":
                    self.status.configure(text="تعذر تنفيذ التنبؤ")
                    self.output_set(payload)
                    self.busy = False
                    self.set_controls(True)
        except queue.Empty:
            pass
        self.root.after(100, self.poll_events)

    def set_controls(self, enabled: bool) -> None:
        state = "normal" if enabled and self.engine is not None and self.image_path is not None and not self.busy else "disabled"
        for button in self.task_buttons.values():
            button.configure(state=state)
        self.all_button.configure(state=state)
        self.pick_button.configure(state="normal" if not self.busy else "disabled")

    def output_set(self, text: str) -> None:
        self.output.configure(state="normal")
        self.output.delete("1.0", "end")
        self.output.insert("1.0", text)
        self.output.tag_add("rtl", "1.0", "end")
        self.output.configure(state="disabled")


def main() -> None:
    root = tk.Tk()
    ModelImageTester(root)
    root.mainloop()


if __name__ == "__main__":
    main()
