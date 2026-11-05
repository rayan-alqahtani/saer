"""إصلاح عقد الصورة في حزمة Kaggle دون تغيير الأوزان أو تنزيل ملفات."""
from __future__ import annotations

import ast
from datetime import datetime, timezone
from pathlib import Path


def repair(root: Path) -> Path:
    path = root / "app.py"
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    for name in ("_valid_image_path", "estimate_image_price", "identify_vehicle"):
        if name not in functions:
            raise RuntimeError(f"توقف: لم أجد الدالة المتوقعة {name}؛ لم أعدل الملف")
    replacements = {
        "_valid_image_path": '''def _valid_image_path(value):
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
''',
        "estimate_image_price": '''def estimate_image_price(upload):
    paths = upload if isinstance(upload, list) else [upload]
    if len(paths) != 1:
        raise ValueError("ارفع صورة واحدة فقط للسيارة")
    checked = [_valid_image_path(path) for path in paths]
    digest = hashlib.sha256("|".join(services.file_sha256(path) for path in checked).encode()).hexdigest()
    result = services.predict_image_price(checked, digest)
    notice = "تقدير تجريبي من صورة واحدة؛ لا يرى الممشى أو الحالة وليس سعر بيع مضموناً."
    return {**result, "image_count": 1, "notice_ar": notice, "notice": notice}
''',
    }
    lines = source.splitlines(keepends=True)
    edits = [(functions[name].lineno - 1, functions[name].end_lineno, code + "\n") for name, code in replacements.items()]
    found = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Call):
            continue
        names = [target.id for target in node.targets if isinstance(target, ast.Name)]
        for name in ("identity_file", "price_files"):
            if name not in names:
                continue
            if ast.unparse(node.value.func) != "gr.File":
                raise RuntimeError(f"توقف: {name} ليس gr.File")
            found.add(name)
            node.value.keywords = [kw for kw in node.value.keywords if kw.arg not in {"file_count", "label"}]
            node.value.keywords.extend([ast.keyword(arg="file_count", value=ast.Constant("single")), ast.keyword(arg="label", value=ast.Constant("صورة السيارة"))])
            indent = " " * node.col_offset
            edits.append((node.lineno - 1, node.end_lineno, indent + ast.unparse(node) + "\n"))
    if found != {"identity_file", "price_files"}:
        raise RuntimeError("توقف: لم أجد مدخلي Gradio المتوقعين؛ لم أعدل الملف")
    for start, end, code in sorted(edits, reverse=True):
        lines[start:end] = [code]
    updated = "".join(lines)
    compile(updated, str(path), "exec")
    backup = path.with_name("app.before-single-image-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f") + ".py")
    backup.write_bytes(path.read_bytes())
    path.write_text(updated, encoding="utf-8")
    print("تم إصلاح مدخلي الصورة، وحفظ النسخة السابقة:", backup)
    return backup


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("bundle_root", type=Path)
    repair(parser.parse_args().bundle_root)
