"""يشغّل الاختبارات ويسجل النتيجة الفعلية، ويختبر .gitignore في مستودع مؤقت منفصل."""
from __future__ import annotations
import io
import json
import shutil
import subprocess
import sys
import unittest
import uuid
from day20_common import *

def check_gitignore():
    git = shutil.which("git")
    if not git:
        return {"passed": False, "error": "git_not_available"}
    repo = ROOT / "tmp" / ("gitignore-check-" + uuid.uuid4().hex[:10])
    repo.mkdir(parents=True, exist_ok=False)
    subprocess.run([git, "init", "--quiet", str(repo)], check=True, capture_output=True)
    shutil.copy2(ROOT / ".gitignore", repo / ".gitignore")
    blocked = ["external_group_a/member1_japanese.csv", "external_group_b/images/a.webp", "external_group_c/prepared/listings.csv", "external_data_bundle.zip", "data/raw/x.csv", "data/processed/listings.csv", "data/recovery/images/a.webp", "models/price.joblib", ".env", ".env.local", "credentials.json", "photo.webp", "book.xlsx", "model.npz", "frontend/node_modules/pkg/index.js"]
    allowed = ["README.md", "docs/DATA_POLICY.md", "scripts/day20_common.py", "tests/test_day20_quality.py", "requirements-data.txt", ".env.example"]
    # LF بايتات صريحة: تحويل stdin إلى CRLF على Windows يجعل Git يرى CR جزءًا من المسار.
    result = subprocess.run([git, "-C", str(repo), "check-ignore", "--no-index", "--stdin"], input=("\n".join(blocked + allowed) + "\n").encode("utf-8"), capture_output=True, check=False)
    ignored = set(result.stdout.decode("utf-8").splitlines())
    return {"passed": result.returncode in (0, 1) and set(blocked) <= ignored and not set(allowed) & ignored, "checked_paths": len(blocked) + len(allowed), "missing_protection": sorted(set(blocked) - ignored), "wrongly_ignored": sorted(set(allowed) & ignored), "test_repo": relative(repo), "git_returncode": result.returncode, "stderr": result.stderr.decode("utf-8", "replace"), "no_project_git_staging_or_push": True}

def run():
    sys.stdout.reconfigure(encoding="utf-8")
    suite = unittest.defaultTestLoader.discover(str(ROOT / "tests"))
    buffer = io.StringIO()
    result = unittest.TextTestRunner(stream=buffer, verbosity=2).run(suite)
    print(buffer.getvalue())
    gitignore = check_gitignore()
    report = {"checked_at": now(), "tests_run": result.testsRun, "failures": len(result.failures), "errors": len(result.errors), "skipped": len(result.skipped), "passed": result.wasSuccessful() and not result.skipped and gitignore["passed"], "gitignore": gitignore, "log": buffer.getvalue()}
    save_json(AUDIT / "unit_tests.json", report)
    print(json.dumps({k: v for k, v in report.items() if k != "log"}, ensure_ascii=False))
    return report["passed"]

if __name__ == "__main__":
    raise SystemExit(0 if run() else 1)
