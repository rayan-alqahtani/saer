"""تطبيق مراجعات البيانات في نسخة مرحلية؛ النشر مشروط بالفحوص ولا يبدأ أي نموذج."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from pathlib import Path
import shutil
from day20_common import *
from day20_data_policy import assign_recognition, assign_make_recognition, yes, reviewed_overview
from day20_content_policy import labels, price_eligibility, assign_attribute_task, support_tier

REVIEW = AUDIT / "content_review"
FINAL = ROOT / "data/staging/day20_content_final"


def resolve_exact_review():
    """تصحيح حالتين بعد تكبير الصورتين المطابقتين: كلتاهما إطار سيارة فقط."""
    hashes = {"cc27c07fb5412771daf653304b28fea6d7ffef5fdde6296378ff59553baba7bf",
              "0176620eccea78c88c1f5480a0722be21aa632ef7fe20fdb9931f8c104568839"}
    changes = []
    for path in sorted((REVIEW / "decisions").glob("*.csv")):
        rows = read_csv(path)
        for row in rows:
            if row["sha256"] in hashes and row["content_kind"] != "vehicle_detail":
                changes.append({"image_uid": row["image_uid"], "sha256": row["sha256"],
                                "before": row["content_kind"], "after": "vehicle_detail"})
                row.update(content_kind="vehicle_detail", review_method="manual_full_image_exact_duplicate_followup", reviewed_at=now())
        save_csv(path, rows)
    if changes:
        save_json(REVIEW / "exact_review_corrections.json", {"checked_at": now(), "changes": changes,
                  "reason_ar": "تم فتح الأصلين بالتكبير؛ كلاهما إطار وجنط فقط وليس منظر السيارة."})


def baseline_guard(baseline_dir=REVIEW):
    index = load_json(baseline_dir / "index.json")
    if any(sha256(PROCESSED / name) != digest for name, digest in index["baseline_sha256"].items()):
        raise ValueError("البيانات تغيرت منذ مراجعة الصور؛ أوقف التطبيق حتى مراجعة الفروق")


def build():
    baseline_guard()
    if FINAL.exists():
        raise ValueError("النسخة المرحلة موجودة؛ لا تستبدل نتائج سابقة دون مراجعة")
    resolve_exact_review()
    listings = read_csv(PROCESSED / "listings.csv")
    images = read_csv(PROCESSED / "images.csv")
    decisions = {}
    for p in sorted((REVIEW / "decisions").glob("*.csv")):
        for row in read_csv(p):
            if row["image_uid"] in decisions: raise ValueError("قرار صورة مكرر")
            decisions[row["image_uid"]] = (row, relative(p))
    if set(decisions) != {im["image_uid"] for im in images}:
        raise ValueError("المراجعة لا تغطي كل الصور بالضبط")
    body = {}
    for p in sorted((REVIEW / "body_decisions").glob("*.csv")):
        for row in read_csv(p):
            if row["listing_uid"] in body: raise ValueError("قرار هيكل مكرر")
            body[row["listing_uid"]] = (row, relative(p))
    if set(body) != {r["listing_uid"] for r in read_csv(REVIEW / "body_index.csv")}:
        raise ValueError("مراجعة الهيكل غير مكتملة")
    sha_kinds = defaultdict(set)
    for im in images:
        decision, path = decisions[im["image_uid"]]
        if decision["sha256"] != im["sha256"] or decision["review_status"] != "reviewed":
            raise ValueError("قرار قديم أو صورة غير مراجعة")
        im.update(content_kind=decision["content_kind"], content_review_status="reviewed",
                  content_review_evidence=path, content_reviewed_at=decision["reviewed_at"])
        sha_kinds[im["sha256"]].add(im["content_kind"])
    if any(len(kinds) != 1 for kinds in sha_kinds.values()):
        raise ValueError("وسم محتوى متعارض للصورة نفسها")
    parents = {r["listing_uid"]: r for r in listings}
    body_changes = []
    for row in listings:
        visual, path = body.get(row["listing_uid"], (None, ""))
        row.update(labels(row, visual))
        row["body_label_evidence"] = path or json.loads(row["field_provenance"]).get("body_type", {}).get("path", "")
        row["color_label_evidence"] = json.loads(row["field_provenance"]).get("color", {}).get("path", "")
        row["usable_price"], row["price_status"] = price_eligibility(row)
        if visual is not None:
            before = labels(row)["body_type_std"]
            if before != row["body_type_std"]:
                body_changes.append({"listing_uid": row["listing_uid"], "source_body_type": row["body_type"],
                                     "before_family": before, "body_type_std": row["body_type_std"], "evidence": path})
    for im in images:
        row = parents[im["listing_uid"]]
        im.update(label_color_std=row["color_std"], label_body_type_std=row["body_type_std"],
                  color_label_status=row["color_label_status"], body_label_status=row["body_label_status"])
    model_support, model_labels = assign_recognition(listings, images)
    make_support, make_labels = assign_make_recognition(listings, images)
    for row in model_support:
        row["support_tier"] = support_tier({s: row["exterior_" + s + "_groups"] for s in ("train", "val", "test")})
    for row in make_support:
        row["support_tier"] = support_tier({s: row[s + "_exterior_groups"] for s in ("train", "val", "test")})
    attribute_support = assign_attribute_task(listings, images, "color", "label_color_std")
    attribute_support += assign_attribute_task(listings, images, "body_type", "label_body_type_std")
    overview_counts = Counter(im["listing_uid"] for im in images if reviewed_overview(im))
    search_parents = {im["listing_uid"] for im in images if yes(im["usable_visual_search"])}
    for row in listings:
        row["n_overview_images"] = overview_counts[row["listing_uid"]]
        row["usable_visual_search"] = row["listing_uid"] in search_parents
    prices = [r for r in listings if r["usable_price"]]
    price_groups = defaultdict(lambda: defaultdict(set))
    for row in prices: price_groups[row["rec_label"]][row["split"]].add(row["group_id"])
    price_support = [{"label": label, **{s + "_groups": len(g[s]) for s in ("train", "val", "test")},
                      "training_support_status": "at_least_5_train_groups" if len(g["train"]) >= 5 else "sparse_training_coverage"}
                     for label, g in sorted(price_groups.items())]
    # نسخ جداول العزل وفهارس التقسيم كما هي، ثم كتابة المخرجات المشتقة فقط.
    shutil.copytree(PROCESSED, FINAL)
    for name, rows in {"listings.csv": listings, "images.csv": images, "price_dataset.csv": prices,
                       "recognition_candidates.csv": images, "class_support.csv": model_support,
                       "make_support.csv": make_support, "attribute_support.csv": attribute_support,
                       "price_support.csv": price_support,
                       "price_exclusions.csv": [r for r in listings if r["condition"] == "used" and not r["usable_price"]]}.items():
        save_csv(FINAL / name, rows, list(listings[0]) if name == "price_exclusions.csv" else None)
    for task, name in [("recognition", "recognition_images"), ("make_recognition", "make_recognition_images"),
                       ("color", "color_images"), ("body_type", "body_images"), ("visual_search", "visual_search_images")]:
        save_csv(FINAL / (name + ".csv"), [im for im in images if yes(im["usable_" + task])], list(images[0]))
    vocab = []
    for field in ("make", "model", "city", "transmission", "type_fuel", "condition", "color", "body_type", "color_std", "body_type_std", "trim_std"):
        vocab.extend({"field": field, "value": v, "listing_count": n, "state": "known" if v else "missing_optional"}
                     for v, n in sorted(Counter(r[field] for r in listings).items()))
    save_csv(FINAL / "categorical_vocabulary.csv", vocab)
    schema = load_json(FINAL / "schema.json")
    schema.update(version="day20-4-content-reviewed", updated_at=now(),
                  all_listing_columns=list(listings[0]), all_image_columns=list(images[0]),
                  content_review_required=True, color_family_map=__import__("day20_content_policy").COLOR_FAMILIES,
                  body_family_map=__import__("day20_content_policy").BODY_FAMILIES,
                  image_labels="قيم المصدر محفوظة. أهلية كل مهمة تتطلب محتوى vehicle_overview مراجعًا؛ الوثائق والداخلية والتفاصيل تبقى للعرض والأرشفة فقط.",
                  optional_labels_policy="الوسوم المشتقة مستقلة. تصحيح الهيكل البصري له دليل؛ اللون عائلة من وصف المصدر وليس تحققًا لونيًا ميدانيًا. petrol يعزل لغموضه.",
                  price_scope="used_only_excluding_flagged_unconfirmed_mileage",
                  evaluation_support_policy="12 مجموعة إجمالًا، 8 تدريب، 1 تحقق، 1 اختبار حد تجريبي فقط. larger_evaluation_sample يعني 20 تدريب و5 تحقق و5 اختبار على الأقل ولا يثبت الدقة.")
    schema["price_features_allowlist"] = ["trim_std" if f == "trim" else f for f in schema["price_features_allowlist"]]
    schema["forbidden_model_features"] = sorted(set(schema["forbidden_model_features"]) |
        (set(listings[0]) - set(schema["price_features_allowlist"])) |
        (set(images[0]) - set(schema["price_features_allowlist"])))
    save_json(FINAL / "schema.json", schema)
    summary = load_json(FINAL / "quality_report.json")
    summary.pop("validation", None)
    summary.update(updated_at=now(), day20_status="awaiting_content_validation", price_rows=len(prices),
                   recognition_ready_images=sum(yes(im["usable_recognition"]) for im in images), supported_model_labels=len(model_labels),
                   supported_make_labels=len(make_labels), make_recognition_ready_images=sum(yes(im["usable_make_recognition"]) for im in images),
                   content_review={"reviewed_images": len(images), "kinds": dict(Counter(im["content_kind"] for im in images)),
                                   "body_reviewed_listings": len(body), "body_derived_changes": len(body_changes),
                                   "body_changes": body_changes, "review_baseline_sha256": load_json(REVIEW / "index.json")["baseline_sha256"]},
                   task_images={task: sum(yes(im["usable_" + task]) for im in images) for task in ("recognition", "make_recognition", "color", "body_type", "visual_search")},
                   price_exclusion_statuses=dict(Counter(r["price_status"] for r in listings if r["condition"] == "used" and not r["usable_price"])),
                   model_support_tiers=dict(Counter(r["support_tier"] for r in model_support)),
                   price_split=dict(Counter(r["split"] for r in prices)),
                   source_optional_missing={f: sum(not r[f] for r in listings) for f in ("color", "body_type", "listing_date")},
                   derived_optional_missing={f: sum(not r[f] for r in listings) for f in ("color_std", "body_type_std")})
    save_json(FINAL / "quality_report.json", summary)
    save_json(REVIEW / "build.json", {"created_at": now(), "stage": relative(FINAL), "training_started": False,
              "listings": len(listings), "images": len(images), "task_images": summary["task_images"]})
    print({"stage": str(FINAL), "task_images": summary["task_images"], "price_rows": len(prices)})


def checked_publication_context(stage_dir, baseline_dir):
    stage_dir, baseline_dir = stage_dir.resolve(), baseline_dir.resolve()
    staging_root, audit_root = ROOT / "data/staging", ROOT / "data/audit"
    if stage_dir == staging_root or baseline_dir == audit_root or not stage_dir.is_relative_to(staging_root) or not baseline_dir.is_relative_to(audit_root):
        raise ValueError("مسار مرحلة أو تجميد خارج مجلدات العمل")
    return stage_dir, baseline_dir


def publish(bundle_dir, stage_dir=FINAL, baseline_dir=REVIEW):
    stage_dir, baseline_dir = checked_publication_context(stage_dir, baseline_dir)
    baseline_guard(baseline_dir)
    report = load_json(AUDIT / "validation.json", {})
    units = load_json(AUDIT / "unit_tests.json", {})
    if not report.get("passed") or report.get("processed_data_path") != relative(stage_dir) or not units.get("passed"):
        raise ValueError("يلزم تحقق مستقل كامل واختبارات ناجحة للنسخة المرحلة قبل النشر")
    # لا نشر لنسخة تغيرت منذ التحقق.
    manifest = load_json(AUDIT / "content_validation_manifest.json", {})
    if not manifest or any(sha256(stage_dir / n) != h for n, h in manifest.items()):
        raise ValueError("النسخة المرحلة تختلف عن بصمات التحقق")
    from datetime import datetime, timezone
    from day20_publish_enriched import checked_move, replace_available
    from verify_essential_bundle import verify
    compact = ROOT / "الملفات_المهمة"
    bundle_dir = bundle_dir.resolve()
    if not bundle_dir.is_relative_to(ROOT / "data/staging") or bundle_dir == ROOT / "data/staging":
        raise ValueError("حزمة النشر يجب أن تكون مرحلية داخل المشروع")
    verify(bundle_dir, stage_dir)
    # فحص تعديل المستخدم في النسخة المختصرة حسب أعمدتها القديمة قبل اعتماد المخطط الجديد.
    from build_essential_bundle import image_path
    for name, source in (("الداتا_المحدثة.csv", "listings.csv"), ("ربط_الصور.csv", "images.csv")):
        current = read_csv(compact / name)
        expected = [{k: image_path(row) if source == "images.csv" and k == "relative_path" else row[k] for k in current[0]}
                    for row in read_csv(baseline_dir / ("baseline_" + source))]
        if current != expected: raise ValueError("تعديل في ملفات المستخدم يحتاج مراجعة قبل النشر")
    replace_available([p for p in PROCESSED.rglob("*") if p.is_file()] + [compact / n for n in ("الداتا_المحدثة.csv", "ربط_الصور.csv")])
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
    archive = ROOT / "data/history" / ("before_content_cleanup_" + stamp)
    pending = ROOT / "data/staging" / ("content_publish_" + stamp) / "processed"
    shutil.copytree(stage_dir, pending)
    checked_move(PROCESSED, archive / "processed")
    try:
        checked_move(pending, PROCESSED)
    except Exception:
        checked_move(archive / "processed", PROCESSED)
        raise
    try:
        checked_move(compact, archive / "الملفات_المهمة")
        try:
            checked_move(bundle_dir, compact)
        except Exception:
            checked_move(archive / "الملفات_المهمة", compact)
            raise
    except Exception:
        checked_move(PROCESSED, pending)
        checked_move(archive / "processed", PROCESSED)
        raise
    compact_check = verify(compact)
    save_json(AUDIT / "content_publication.json", {"published_at": now(), "archive": relative(archive),
              "processed_files_sha256": {p.relative_to(PROCESSED).as_posix(): sha256(p) for p in PROCESSED.rglob("*") if p.is_file()},
              "compact_sync": compact_check, "training_started": False})
    save_json(AUDIT / "compact_bundle.json", {"stage": str(compact), "processed_data_path": str(PROCESSED), "published": True,
              "counts": compact_check, "verified_at": now(), "previous_bundle_archive": str(archive / "الملفات_المهمة")})
    print({"published": True, "archive": str(archive), "compact_sync": compact_check})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("build", "publish"))
    parser.add_argument("--bundle-dir", type=Path)
    parser.add_argument("--stage-dir", type=Path, default=FINAL)
    parser.add_argument("--baseline-dir", type=Path, default=REVIEW)
    args = parser.parse_args()
    if args.action == "build": build()
    elif args.bundle_dir: publish(args.bundle_dir, args.stage_dir, args.baseline_dir)
    else: parser.error("يلزم --bundle-dir للنشر")
