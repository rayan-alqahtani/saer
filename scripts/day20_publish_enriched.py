"""نشر النسخة المجربة والحزمة المختصرة مع أرشفة السابقة وفحوص نهائية فعلية."""
import argparse
import ctypes
from ctypes import wintypes
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shutil
import sys

from day20_common import ROOT, AUDIT, PROCESSED, STAGING, load_json, save_json, save_bytes, sha256, now, read_csv


def checked_move(source, destination):
    source, destination = source.resolve(), destination.resolve()
    if not source.is_relative_to(ROOT) or not destination.is_relative_to(ROOT) or ROOT in {source, destination}:
        raise ValueError("نقل خارج حدود مجلد المشروع")
    if source.is_relative_to(ROOT / "data/raw") or destination.is_relative_to(ROOT / "data/raw"):
        raise ValueError("الخام خارج نطاق النشر")
    if not source.is_dir() or destination.exists() or destination.is_relative_to(source):
        raise ValueError("مصدر أو وجهة النقل غير آمنين")
    destination.parent.mkdir(parents=True, exist_ok=True)
    source.rename(destination)


def replace_available(paths):
    if os.name != "nt":
        return
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, wintypes.LPVOID, wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
    kernel.CreateFileW.restype = wintypes.HANDLE
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    locked = []
    for path in paths:
        if not path.exists():
            continue
        handle = kernel.CreateFileW(str(path), 0x10000, 7, None, 3, 128, None)
        if handle == wintypes.HANDLE(-1).value:
            locked.append({"path": str(path), "windows_error": ctypes.get_last_error()})
        else:
            kernel.CloseHandle(handle)
    if locked:
        raise PermissionError("ملفات غير متاحة للاستبدال الآمن: " + json.dumps(locked, ensure_ascii=False))


def publish(stage_root, bundle_root):
    stage_root, bundle_root = stage_root.resolve(), bundle_root.resolve()
    if not all(path.is_relative_to(ROOT / "data/staging") and path != ROOT / "data/staging" for path in [stage_root, bundle_root]):
        raise ValueError("مصادر النشر يجب أن تكون نسخًا مرحلية داخل المشروع")
    ready = load_json(stage_root / "stage_ready.json")
    if not ready or not ready["validation"]["passed"] or not ready["changed_fields_source_verified"]:
        raise ValueError("لا نشر قبل نجاح فحص المرحلة وأصل تغييراتها")
    staged_processed = stage_root / "processed"
    if any(sha256(staged_processed / relative) != digest for relative, digest in ready["processed_files"].items()):
        raise ValueError("تغير ملف من النسخة المرحلة بعد فحصها")
    for current, baseline in [("listings.csv", "enrichment_baseline_listings.csv"), ("images.csv", "enrichment_baseline_images.csv"), ("quarantine_listings.csv", "enrichment_baseline_quarantine_listings.csv")]:
        if sha256(PROCESSED / current) != sha256(AUDIT / baseline):
            raise ValueError("تغير ملف المستخدم أثناء الاستكمال؛ يلزم مراجعة التعديل: " + current)
    from build_essential_bundle import LISTING_FILE, IMAGE_FILE, image_path
    essential = ROOT / "الملفات_المهمة"
    if {p.name for p in essential.iterdir()} != {LISTING_FILE, IMAGE_FILE, "الصور"}:
        raise ValueError("تغير محتوى المجلد المختصر؛ يلزم مراجعة الإضافة أولًا")
    for current_name, baseline_name in [(LISTING_FILE, "enrichment_baseline_listings.csv"), (IMAGE_FILE, "enrichment_baseline_images.csv")]:
        actual = read_csv(essential / current_name)
        baseline_rows = read_csv(AUDIT / baseline_name)
        fields = list(actual[0])
        expected = [{field: image_path(row) if current_name == IMAGE_FILE and field == "relative_path" else row[field] for field in fields} for row in baseline_rows]
        if actual != expected:
            raise ValueError("تغيرت قيم المجلد المختصر أثناء العمل؛ يلزم حفظ تعديل المستخدم ومراجعته")
    from verify_essential_bundle import verify
    verify(bundle_root, staged_processed)
    audit_outputs = ["issues.csv", "changes.csv", "duplicate_pairs.csv", "structured_duplicates.csv", "image_exclusions.csv", "source_gates.json", "recovered_image_stats.csv", "price_outlier_flags.csv", "global_price_iqr_flags.csv", "enrichment_review.json", "external_analysis_final_review.json"]
    target_files = list(PROCESSED.rglob("*")) + [essential / LISTING_FILE, essential / IMAGE_FILE]
    target_files += [AUDIT / name for name in audit_outputs]
    replace_available([path for path in target_files if not path.is_dir()])
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
    archive = ROOT / "data/history" / ("before_full_enrichment_" + stamp)
    pending = ROOT / "data/staging" / ("publish_ready_" + stamp) / "processed"
    transaction = {"started_at": now(), "phase": "verified_before_activation", "stage_root": str(stage_root), "bundle_root": str(bundle_root), "archive": str(archive), "pending_processed": str(pending)}
    save_json(AUDIT / "enrichment_publication_transaction.json", transaction)
    shutil.copytree(staged_processed, pending)
    checked_move(PROCESSED, archive / "processed")
    try:
        checked_move(pending, PROCESSED)
    except Exception:
        checked_move(archive / "processed", PROCESSED)
        raise
    try:
        checked_move(essential, archive / "الملفات_المهمة")
        try:
            checked_move(bundle_root, essential)
        except Exception:
            checked_move(archive / "الملفات_المهمة", essential)
            raise
    except Exception:
        checked_move(PROCESSED, pending)
        checked_move(archive / "processed", PROCESSED)
        raise
    transaction.update(phase="activated_awaiting_final_checks", activated_at=now())
    save_json(AUDIT / "enrichment_publication_transaction.json", transaction)
    for name in audit_outputs:
        save_bytes(AUDIT / name, (stage_root / "audit" / name).read_bytes())
    for path in (stage_root / "sources").rglob("*"):
        if path.is_file():
            save_bytes(STAGING / path.relative_to(stage_root / "sources"), path.read_bytes())
    import day20_validate
    validation = day20_validate.validate()
    if not validation["passed"]:
        raise ValueError("فشل تحقق النسخة المنشورة؛ لا إعلان اكتمال")
    from day20_enrichment_audit import audit
    review = audit(PROCESSED, AUDIT)
    import run_checks
    if not run_checks.run():
        raise ValueError("فشلت الاختبارات النهائية؛ لا إعلان اكتمال")
    compact = verify(essential)
    save_json(AUDIT / "compact_bundle.json", {"stage": str(essential), "processed_data_path": str(PROCESSED), "published": True, "counts": compact, "verified_at": now(), "previous_bundle_archive": str(archive / "الملفات_المهمة")})
    old_review = load_json(AUDIT / "external_analysis_review.json")
    old_review.update(historical_snapshot=True, superseded_by="data/audit/day20/external_analysis_final_review.json", previous_processed_archive=str(archive / "processed"))
    save_json(AUDIT / "external_analysis_review.json", old_review)
    import day20_report
    day20_report.report()
    quality = load_json(PROCESSED / "quality_report.json")
    readme_path = ROOT / "README.md"
    readme = readme_path.read_text(encoding="utf-8")
    readme = re.sub(r"(\[الإعلانات المنظفة\]\(data/processed/listings.csv\): )\d+", lambda m: m[1] + str(quality["clean_listings"]), readme)
    readme = re.sub(r"(\[فهرس الصور المعتمدة\]\(data/processed/images.csv\): )\d+", lambda m: m[1] + str(quality["clean_images"]), readme)
    readme = readme.replace("الحجم المقاس في 20 أكتوبر نحو 1.73 جيجابايت.", "كان الحجم المقاس قبل الاستكمال الموسع في 20 أكتوبر نحو 1.73 جيجابايت. أضيفت لاحقًا أدلة المصدر المضغوطة ونسخ الحفظ الآمنة؛ الأحجام التالية تخص الجرد السابق.")
    save_bytes(readme_path, readme.encode("utf-8"))
    units = load_json(AUDIT / "unit_tests.json")
    progress_path = ROOT / "PROGRESS.md"
    progress = progress_path.read_text(encoding="utf-8")
    progress += f"""\n## إغلاق الاستكمال الموسع للبيانات — 20 أكتوبر 2026

- **ما أنجزته:** بعد موافقة المستخدم فحصت كل حالات الاستكمال الـ{quality['enrichment']['queue_size']}، وراجعت ملاحظات التحليل الـ25 على النسخة الحالية. اعتمدت {quality['clean_listings']} إعلانًا و{quality['clean_images']} صورة، مع عزل موثق لـ{quality['quarantined_listings']} إعلانًا. حفظت قيم السعر والممشى التاريخية، وفصلت مراجعة الممشى الحالية. نشرت الجدولين والصور فقط في الملفات_المهمة بعد فحصهما، وأرشفت النسخة السابقة كاملة في {archive.relative_to(ROOT).as_posix()}.
- **ما اختبرته والنتيجة:** {validation['passed_count']}/{validation['total_checks']} فحص جودة على المسار النهائي؛ {units['tests_run']} اختبارًا آليًا بلا أخطاء أو تخطٍّ؛ تحقق بصمات الأصول والخام والصور، ومطابقة HTML وJSON وهوية كل دليل؛ مطابقة الحزمة المختصرة وقيمها وصورها دون زيادة أو نقص؛ كل حقل تغير له أصل مطابق.
- **الملفات المتغيرة:** data/processed، الملفات_المهمة، data/staging/day20، سجلات التدقيق والاسترجاع، scripts وtests، docs وREADME وPROGRESS. لم تعدل الأصول أو raw، ولم يحدث تدريب أو رفع Git.
- **المشكلات:** نتائج المصدر: {quality['enrichment']['status_counts']}. تغطية الحقول قبل وبعد والباقي في enrichment_review.json. القيم غير المذكورة في المصدر وصفحاته وصوره غير المتاحة تبقى بحالة صريحة؛ لا صور بديلة ولا حقائق مختلقة. تأكيد الممشى المنشور لا يعني فحص العداد ميدانيًا.
- **الأوامر:** day20_enrich.py --limit 2500 --budget-mb 230؛ day20_enrich_images.py؛ day20_stage_enriched.py؛ build_essential_bundle.py --processed-dir؛ day20_publish_enriched.py. النشر استدعى Validate وaudit وrun_checks وفحص الحزمة ثم Report فعليًا.
- **الخطوة التالية:** عمل 21 أكتوبر حسب ROADMAP: baseline سعر المستعمل وembeddings وفهرس التشابه، ثم بوابة M1 في 22 أكتوبر. لا واجهة قبل أساس البيانات والنماذج القابل للاختبار.
- **هل تحتاج موافقتي؟:** لا لهذه المهمة؛ الاستكمال الكامل مصرح به صراحة، والنشر محلي محفوظ النسخ. لا يشمل ذلك بدء تدريب طويل أو تنزيل أوزان كبيرة.
"""
    save_bytes(progress_path, progress.encode("utf-8"))
    result = {"published_at": now(), "passed": True, "archive": str(archive), "stage_root": str(stage_root), "validation": quality["validation"], "unit_tests": units["tests_run"], "compact": compact, "all_changed_fields_source_verified": review["all_changed_fields_source_verified"]}
    save_json(AUDIT / "enrichment_publication.json", result)
    transaction.update(phase="complete", completed_at=now(), final_checks_passed=True)
    save_json(AUDIT / "enrichment_publication_transaction.json", transaction)
    state = load_json(AUDIT / "enrichment_work_state.json", {})
    state.update(active_collection_session=None, published=True, completed_at=now(), remaining_steps=[], publication=result)
    save_json(AUDIT / "enrichment_work_state.json", state)
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return result


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-root", type=Path, default=ROOT / "data/staging/day20_enriched")
    parser.add_argument("--bundle-root", type=Path, required=True)
    args = parser.parse_args()
    publish(args.stage_root, args.bundle_root)
