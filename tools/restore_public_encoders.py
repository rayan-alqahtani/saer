"""استعادة مشفرات عامة مثبتة البصمات؛ لا أسرار أو تدريب أو استبدال ملفات موجودة."""
from pathlib import Path
import argparse
import hashlib
import io
import json
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]

def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()

def download(url, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_suffix(path.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "saer-delivery/1"})
    with urllib.request.urlopen(req, timeout=120) as response, part.open("wb") as stream:
        while block := response.read(1024 * 1024):
            stream.write(block)
    return part

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true", help="تنزيل الملفات المفقودة فقط، نحو 2.72 GiB")
    parser.add_argument("--home", type=Path, default=Path.home(), help="جذر استعادة للاختبار؛ التشغيل الحالي يقرأ منزل المستخدم الافتراضي")
    parser.add_argument("--only-source", action="store_true", help="فحص أو استعادة شفرة DINOv2 فقط")
    args = parser.parse_args()
    spec = json.loads((ROOT / "docs/PUBLIC_ENCODERS.json").read_text(encoding="utf-8"))
    home = args.home.resolve()
    checked = 0
    if not args.only_source:
        for item in spec["files"]:
            path = home / item["path"]
            if not path.exists():
                if not args.download:
                    raise FileNotFoundError(f"ملف مفقود: {path}; أعد الأمر مع --download")
                part = download(item["url"], path)
                if part.stat().st_size != item["bytes"] or digest(part) != item["sha256"]:
                    raise ValueError(f"بصمة التنزيل غير مطابقة: {part}")
                part.rename(path)
            if path.stat().st_size != item["bytes"] or digest(path) != item["sha256"]:
                raise ValueError(f"ملف موجود ببصمة مختلفة؛ لم يُستبدل: {path}")
            checked += 1
        for relative, revision in spec["refs"].items():
            path = home / relative
            if path.exists() and path.read_text().strip() != revision:
                raise ValueError(f"مرجع موجود مختلف؛ لم يُستبدل: {path}")
            if not path.exists():
                if not args.download: raise FileNotFoundError(path)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(revision, encoding="utf-8")
    source = spec["source"]
    folder = home / source["path"]
    if not folder.exists():
        if not args.download: raise FileNotFoundError(folder)
        archive = download(source["url"], home / ".cache/saer-downloads/dinov2-source.zip")
        if digest(archive) != source["archive_sha256"]: raise ValueError("بصمة أرشيف الشفرة غير مطابقة")
        with zipfile.ZipFile(archive) as z:
            for item in z.infolist():
                parts = Path(item.filename).parts[1:]
                if not parts or item.is_dir(): continue
                target = folder.joinpath(*parts).resolve()
                if not target.is_relative_to(folder.resolve()): raise ValueError("مسار أرشيف غير آمن")
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(z.read(item))
    for relative, expected in source["python_files"].items():
        path = folder / relative
        if not path.is_file() or hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest() != expected:
            raise ValueError(f"شفرة DINOv2 مختلفة أو مفقودة: {path}")
    print(json.dumps({"public_files_verified": checked, "source_python_files_verified": len(source["python_files"]), "source_revision": source["revision"], "download_enabled": args.download}, ensure_ascii=False))

if __name__ == "__main__":
    main()
