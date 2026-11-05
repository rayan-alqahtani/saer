"""اختبارات مستقلة لنتائج إصلاح الملاحظات الست عشرة."""
from collections import Counter
import hashlib
import json
from day20_common import *


def issue_checks(processed, listings, images, schema, summary):
    checks = []
    def check(code, description, okay, detail=""):
        checks.append(("T-U16-" + code, description, bool(okay), detail))
    base = ROOT / summary["user_issues"]["audit_path"]
    old_l = {r["listing_uid"]: r for r in read_csv(base / "baseline_listings.csv")}
    old_i = {r["image_uid"]: r for r in read_csv(base / "baseline_images.csv")}
    mutable_l = {"body_type_std", "body_label_status", "body_label_evidence", "usable_body_type"}
    mutable_i = {"label_body_type_std", "body_label_status", "usable_body_type", "body_type_status", "recognition_status",
                 "make_recognition_status", "color_status", "exterior_status"}
    check("FROZEN", "كل القيم السابقة محفوظة عدا التصحيح المراجع للهيكل والحالات المشتقة",
          set(old_l) == {r["listing_uid"] for r in listings} and set(old_i) == {r["image_uid"] for r in images}
          and all(all(r[k] == v for k, v in old_l[r["listing_uid"]].items() if k not in mutable_l) for r in listings)
          and all(all(r[k] == v for k, v in old_i[r["image_uid"]].items() if k not in mutable_i) for r in images))
    reviews = [r for p in (base / "decisions").glob("*.csv") for r in read_csv(p)]
    expected_ids = {r["listing_uid"] for r in read_csv(base / "visual_index.csv")}
    evidence_ok = True
    by_id = {r["listing_uid"]: r for r in listings}
    for row in reviews:
        actual = {(r["image_uid"], r["sha256"], r["relative_path"]) for r in images if r["listing_uid"] == row["listing_uid"]}
        evidence = {(r["image_uid"], r["sha256"], r["relative_path"]) for r in json.loads(row["evidence_images"])}
        evidence_ok &= evidence == actual and sha256(ROOT / row["board_path"]) == row["board_sha256"]
        if row["identity_review"] == "insufficient_visual_evidence":
            evidence_ok &= not any(r["usable_recognition"] == "True" for r in images if r["listing_uid"] == row["listing_uid"])
    corrected = [r for r in reviews if r["body_visual"]]
    check("BODY-REVIEW", "كل T-Roc وCountryman راجعت صورها وبصماتها وصحح الهدف دون تغيير المصدر",
          len(reviews) == len(expected_ids) == 60 and {r["listing_uid"] for r in reviews} == expected_ids and evidence_ok
          and len(corrected) == 30 and all(by_id[r["listing_uid"]]["body_type_std"] == r["body_visual"] == "suv_crossover" for r in corrected)
          and all(r["body_type_std"] == "suv_crossover" for r in listings if r["model"] in {"T-Roc", "Countryman"}))
    grand = [r for r in listings if r["make"] == "Hyundai" and r["model"] == "Grand i10"]
    check("GRAND-I10", "وسم Grand i10 المشتق سيدان لكل الحالات الثماني والخمسين دون تخمين الحقل الأصلي",
          len(grand) == 58 and all(r["body_type_std"] == "sedan" for r in grand))
    columns = {"flag_source_unavailable": "recovery_source_redirected_or_unavailable", "flag_high_annual_mileage": "annualized_mileage_proxy_high_review",
               "flag_price_iqr_outlier": "global_price_iqr_outlier_review", "flag_mileage_exact_100": "mileage_exact_100_review", "flag_young_low_mileage": "young_used_low_mileage_review"}
    flags_ok = all((r[column] == "True") == (code in set(r["data_quality_flags"].split(";"))) for r in listings for column, code in columns.items())
    check("FLAGS", "الأعلام المنفصلة تطابق كل علامة داخل الخلايا المركبة ولا تدخل ميزات السعر",
          flags_ok and not set(columns) & set(schema["price_features_allowlist"]),
          {column: sum(r[column] == "True" for r in listings) for column in columns})
    check("REVIEW-SEMANTICS", "الصور المستبعدة مراجعة وليست معلقة، ونقص نوع المصدر لا يلغي مراجعة المحتوى",
          all(r["recognition_status"] == r["make_recognition_status"] == r["color_status"] == r["body_type_status"] == "reviewed_not_vehicle_overview"
              for r in images if r["content_kind"] != "vehicle_overview")
          and all((r["exterior_status"] == "source_gallery_kind_missing") == (not r["image_kind"]) for r in images)
          and all(r["content_review_status"] == "reviewed" for r in images))
    route = lambda r: "model_and_make" if r["usable_recognition"] == "True" else "make_only" if r["usable_make_recognition"] == "True" else "unavailable"
    fallback = [r for r in images if route(r) == "make_only"]
    check("FALLBACK", "مسار الماركة للفئات غير المدعومة منفصل ولا يمحو هوية الموديل",
          all(r["recognition_route"] == route(r) for r in listings + images)
          and read_csv(processed / "make_fallback_images.csv") == fallback
          and all(r["rec_label"] == r["make"] + "|" + r["model"] for r in listings),
          {"listing_routes": dict(Counter(r["recognition_route"] for r in listings)), "fallback_images": len(fallback)})
    manifest = load_json(processed / "training_manifest.json")
    hashes = {name: hashlib.sha256((processed / name).read_bytes()).hexdigest() for name in ("listings.csv", "images.csv", "schema.json")}
    fingerprint = hashlib.sha256(json.dumps(hashes, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    folds = sorted((r["listing_uid"], r["group_id"], r["split"]) for r in listings)
    split_hash = hashlib.sha256(json.dumps(folds, ensure_ascii=False, separators=(",", ":")).encode("utf-8")).hexdigest()
    historic = {r["listing_uid"]: r for r in read_csv(AUDIT / "enrichment_baseline_listings.csv")}
    changed = {r["listing_uid"] for r in listings if r["split"] != historic[r["listing_uid"]]["split"]}
    check("VERSION", "بصمات نسخة البيانات والتقسيم مثبتة والتغيير التاريخي 43 ولم يتغير قسم في هذا الإصلاح",
          manifest["dataset_version"] == schema["version"] and manifest["dataset_fingerprint"] == fingerprint
          and manifest["split_fingerprint"] == split_hash and manifest["input_files_sha256"] == hashes
          and all(r["split"] == old_l[r["listing_uid"]]["split"] for r in listings)
          and len(changed) == 43 and changed == {r["listing_uid"] for r in read_csv(base / "historical_split_changes.csv")} )
    q = [r for r in read_csv(processed / "quarantine_listings.csv") if r["source_dataset"] == "external_group_c"]
    reasons = {r["listing_uid"]: r["quarantine_reasons"] for r in read_csv(base / "kg_quarantine_reasons.csv")}
    check("KG", "كل إعلان KG المعزول له سبب ظاهر في سجل المراجعة، ولا أدخلت سجلات المعزولين في التدريب",
          len(q) == 20 and reasons == {r["listing_uid"]: r["quarantine_reasons"] for r in q}
          and all(reasons.values()) and not set(reasons) & set(by_id))
    return checks
