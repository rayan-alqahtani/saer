"""حزمة احتياطية من commit نظيف خارج المستودع، بلا بيئات أو أسرار أو كاش."""
from pathlib import Path, PurePosixPath
import argparse
import hashlib
import io
import json
import re
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[1]

def git(*args):
    return subprocess.check_output(["git", "-C", str(ROOT), *args])

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.is_relative_to(ROOT) or output.exists():
        raise ValueError("اختر مسار ZIP جديداً خارج المستودع؛ لا استبدال لحزمة موجودة")
    if git("status", "--porcelain").strip():
        raise RuntimeError("احفظ الملفات في commit نظيف قبل التحزيم")
    commit = git("rev-parse", "HEAD").decode().strip()
    branch = git("branch", "--show-current").decode().strip()
    contents = git("-c", "core.autocrlf=false", "archive", "--format=zip", "HEAD")
    source = zipfile.ZipFile(io.BytesIO(contents))
    entries = [item for item in source.infolist() if not item.is_dir()]
    forbidden_dirs = {".git", ".runtime", ".runtime-cache", "node_modules", "__pycache__", "dist", "dist-demo", "build"}
    patterns = [rb"gh[pousr]_[A-Za-z0-9]{30,}", rb"github_pat_[A-Za-z0-9_]{30,}", rb"hf_[A-Za-z0-9]{30,}", rb"AKIA[0-9A-Z]{16}", rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"]
    for item in entries:
        path = PurePosixPath(item.filename)
        if forbidden_dirs.intersection(path.parts) or path.name in {".env", ".env.local", "credentials.json"} or item.filename.startswith("data/raw/"):
            raise ValueError(f"مسار محظور: {item.filename}")
        data = source.read(item)
        if path.suffix not in {".joblib", ".sqlite", ".png", ".jpg", ".jpeg", ".webp", ".ico", ".woff", ".woff2"}:
            if any(re.search(pattern, data) for pattern in patterns):
                raise ValueError(f"نمط سر محتمل: {item.filename}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "x", zipfile.ZIP_DEFLATED) as archive:
        for item in entries: archive.writestr(item.filename, source.read(item))
        version = f"نسخة تسليم سعّر\nالفرع: {branch}\ncommit: {commit}\nاقرأ README.md وdocs/DATA_POLICY.md قبل التشغيل. المشفرات العامة غير مرفقة؛ تعليمات الاستعادة داخل الحزمة.\nلم تُعد هذه النسخة نشر الموقع.\n"
        archive.writestr("VERSION.txt", version.encode("utf-8"))
        archive.comment = commit.encode("ascii")
    with zipfile.ZipFile(output) as archive:
        assert archive.testzip() is None
        for item in entries:
            assert archive.read(item.filename) == source.read(item)
        assert archive.read("VERSION.txt").decode("utf-8").find(commit) >= 0
    result = {"commit": commit, "zip_files": len(entries) + 1, "zip_bytes": output.stat().st_size, "sha256": hashlib.sha256(output.read_bytes()).hexdigest(), "crc": "passed", "git_files_byte_comparison": "passed", "secret_pattern_matches": 0}
    output.with_suffix(".sha256").write_text(result["sha256"] + "  " + output.name + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))

if __name__ == "__main__":
    main()
