"""تحقق مستقل من الملفات المحفوظة، مع مطابقة الخام والنسخة والأصل بالهاش."""
from __future__ import annotations
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import json
from datetime import date
import statistics
from pathlib import Path
import sys
from day20_common import *

def validate(check_raw=True):
    listings = read_csv(PROCESSED / "listings.csv")
    images = read_csv(PROCESSED / "images.csv")
    quarantine = read_csv(PROCESSED / "quarantine_listings.csv")
    exclusions = read_csv(PROCESSED / "image_exclusions.csv")
    gates = load_json(AUDIT / "source_gates.json")
    recovery = load_json(RECOVERY / "manifest.json")
    summary = load_json(PROCESSED / "quality_report.json")
    checks = []
    def check(code, description, okay, detail=""):
        record = {"id": code, "description_ar": description, "passed": bool(okay), "detail": detail}
        checks.append(record)
        print(json.dumps(record, ensure_ascii=False), flush=True)
    lid = {r["listing_uid"] for r in listings}
    qid = {r["listing_uid"] for r in quarantine}
    iid = {r["image_uid"] for r in images}
    expected_ids = {uid(source, r["id_listing"]) for source, filename in SOURCES.items() for r in read_csv(RAW / source / filename)}
    by_id = {r["listing_uid"]: r for r in listings}
    check("T-D01a", "كل معرّف إعلان وصورة فريد", len(lid) == len(listings) and len(iid) == len(images) and len(qid) == len(quarantine))
    check("T-D01b", "كل سجل مرتبط بمصدره ومعرّفه الأصلي", all(r["listing_uid"] == uid(r["source_dataset"], r["source_listing_id"]) for r in listings + quarantine))
    check("T-D03a", "لا فقد صامت ولا تداخل بين المعتمد والمعزول", not lid & qid and lid | qid == expected_ids, {"raw": len(expected_ids), "accepted": len(lid), "quarantined": len(qid)})
    check("T-D02a", "كل صورة معتمدة مرتبطة بإعلانها ومعرّف مصدرها المطابق",
          all(r["listing_uid"] in lid and r["source_url"].startswith("https://cdn.example.invalid/")
              and "/posts/" + by_id[r["listing_uid"]]["external_listing_id"] + "/" in r["source_url"] for r in images))
    counts = Counter(r["listing_uid"] for r in images)
    check("T-D02b", "كل إعلان معتمد له صور صحيحة وعددها مطابق", all(counts[r["listing_uid"]] == int(r["n_images"]) and counts[r["listing_uid"]] > 0 for r in listings))
    required = ["make", "model", "year", "sar_price", "condition", "city", "transmission", "type_fuel"]
    check("T-D03b", "لا نواقص أساسية في الجداول المعتمدة", all(all(text(r.get(f)) for f in required) and not r["quarantine_reasons"] for r in listings))
    model_names = defaultdict(set)
    for row in listings:
        model_names[(row["make"], model_key(row["model"]))].add(row["model"])
    check("T-D-MODEL-NAMES", "لا تنقسم تهجئات الموديل المتكافئة وmodel_std مطابق للموديل",
          all(len(names) == 1 for names in model_names.values()) and all(row["model"] == row["model_std"] for row in listings),
          [{"make": key[0], "model_key": key[1], "variants": sorted(names)} for key, names in model_names.items() if len(names) > 1])
    check("T-D03c", "كل سجل معزول له سبب واضح", all(text(r["quarantine_reasons"]) for r in quarantine))
    valid_numbers = all(number(r["year"]) is not None and 2018 <= number(r["year"]) <= 2026 and number(r["sar_price"]) is not None and 8000 <= number(r["sar_price"]) <= 3_000_000 and (r["condition"] == "new" or number(r["km_mileage"]) is not None and 0 <= number(r["km_mileage"]) <= 600000) for r in listings)
    check("T-D03d", "سنوات وأسعار وممشى المعتمد ضمن القواعد الموثقة", valid_numbers)
    def inspect_image(r):
        errors = []
        path = Path(r["relative_path"])
        resolved = (ROOT / path).resolve()
        if path.is_absolute() or not resolved.is_relative_to(ROOT / "data") or not resolved.is_file():
            errors.append({"image_uid": r["image_uid"], "reason": "path_missing_or_unsafe"})
        elif sha256(resolved) != r["sha256"]:
            errors.append({"image_uid": r["image_uid"], "reason": "hash_mismatch"})
        if r["is_valid"] != "True":
            errors.append({"image_uid": r["image_uid"], "reason": "not_validated"})
        return errors
    file_errors = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        for errors in pool.map(inspect_image, images):
            file_errors.extend(errors)
    check("T-D02c", "كل مسار نسبي آمن وموجود والهاش مطابق لفحص الصورة", not file_errors, file_errors[:10])
    folds = {r["listing_uid"]: r["split"] for r in listings}
    group_folds, hash_folds, visual_folds = defaultdict(set), defaultdict(set), defaultdict(set)
    for r in listings:
        group_folds[r["group_id"]].add(r["split"])
    for r in images:
        hash_folds[r["sha256"]].add(r["split"])
        visual_folds[r["duplicate_group"]].add(r["split"])
    check("T-D04a", "صور الإعلان تتبع تقسيم الإعلان نفسه", all(r["split"] == folds[r["listing_uid"]] and r["group_id"] == by_id[r["listing_uid"]]["group_id"] for r in images))
    check("T-D04b", "لا مجموعة إعلان أو صورة مطابقة تعبر أقسام التدريب والاختبار", all(len(v) == 1 for v in group_folds.values()) and all(len(v) == 1 for v in hash_folds.values()) and all(len(v) == 1 for v in visual_folds.values()))
    pairs = read_csv(AUDIT / "duplicate_pairs.csv")
    check("T-D04c", "الأزواج البصرية المشتبه بها في قسم واحد عند اعتماد الطرفين", all(folds[r["left_listing_uid"]] == folds[r["right_listing_uid"]] for r in pairs if r["left_listing_uid"] in folds and r["right_listing_uid"] in folds))
    split_ids = {}
    for split in ["train", "val", "test"]:
        rs = read_csv(PROCESSED / "splits" / (split + ".csv"))
        split_ids[split] = {r["listing_uid"] for r in rs}
    check("T-D04d", "فهارس التقسيم محفوظة وتغطي المعتمد مرة واحدة", set.union(*split_ids.values()) == lid and sum(len(v) for v in split_ids.values()) == len(lid) and all(all(folds[k] == split for k in keys) for split, keys in split_ids.items()))
    source_gate_times = [g["checked_at"] for g in gates.values()]
    check("T-D-GATE", "الدمج جرى بعد نجاح المصادر وانتهاء الاسترجاع", all(g["passed"] for g in gates.values()) and recovery.get("complete") and recovery.get("fallback_complete") and all(t <= summary["created_at"] for t in source_gate_times))
    check("T-D-IMG-COUNT", "كل ملف صورة خام أو مسترجع معتمد أو معزول", len(images) + len(exclusions) == summary["raw_images"] + summary["recovered_images"], {"accepted": len(images), "excluded": len(exclusions), "input": summary["raw_images"] + summary["recovered_images"]})
    issue_rows = read_csv(AUDIT / "issues.csv")
    check("T-D-ISSUES", "كل مشكلة محصورة لها إصلاح أو عزل أو توثيق وفق العقد", all(r["action"] in {"fixed", "quarantined", "documented"} and r["evidence"] for r in issue_rows))
    schema = load_json(PROCESSED / "schema.json")
    check("T-D-LEAK-FEATURES", "سمات السعر لا تتضمن الهدف أو المعرّفات أو المسارات", not set(schema["price_features_allowlist"]) & (set(schema["forbidden_model_features"]) | {"sar_price"}))
    check("T-D-PROVENANCE", "سجل مصدر الحقول موجود وقابل للقراءة", all(isinstance(json.loads(r["field_provenance"]), dict) and isinstance(json.loads(r["field_states"]), dict) and (ROOT / r["raw_metadata_path"]).is_file() for r in listings + quarantine))
    check("T-D-AVAILABILITY", "لا ادعاء توفر حي للإعلانات", all(r["live_availability_verified"] == "False" for r in listings))
    photo_labels = defaultdict(set)
    for r in images:
        if r["usable_recognition"] == "True":
            photo_labels[r["sha256"]].add(r["rec_label"])
    content_policy = schema.get("content_review_required", False)
    def eligible_overview(r):
        return (r.get("content_kind") == "vehicle_overview" and r.get("content_review_status") == "reviewed") if content_policy else r["is_exterior"] == "True"
    check("T-D-REC", "مجموعة التعرف المهيأة مناظر سيارة مراجعة وفئات مدعومة دون تكرار", all(eligible_overview(r) and r["rec_label"] != "other" for r in images if r["usable_recognition"] == "True") and sum(r["usable_recognition"] == "True" for r in images) == len(photo_labels))
    exterior_groups = defaultdict(lambda: defaultdict(set))
    candidate_ids, seen_exterior = set(), set()
    for r in sorted(images, key=lambda x: x["image_uid"]):
        if eligible_overview(r) and r["sha256"] not in seen_exterior:
            seen_exterior.add(r["sha256"])
            candidate_ids.add(r["image_uid"])
            exterior_groups[r["rec_label"]][r["split"]].add(r["group_id"])
    policy = schema["recognition_policy"]
    expected_supported = {label for label, groups in exterior_groups.items()
                          if sum(len(v) for v in groups.values()) >= policy["min_total_groups"]
                          and all(len(groups[s]) >= policy["min_" + s + "_groups"] for s in ["train", "val", "test"])}
    selected = [r for r in images if r["usable_recognition"] == "True"]
    expected_selected = {r["image_uid"] for r in images if r["image_uid"] in candidate_ids and r["rec_label"] in expected_supported}
    reported_supported = {r["label"] for r in read_csv(PROCESSED / "class_support.csv") if r["supported"] == "True"}
    check("T-D-REC-SUPPORT", "فئات التعرف تحقق دعم الصور الخارجية الفريدة في التدريب والتحقق والاختبار",
          {r["image_uid"] for r in selected} == expected_selected
          and reported_supported == expected_supported
          and summary["supported_model_labels"] == len(expected_supported),
          {"supported_labels": sorted(expected_supported), "selected_images": len(selected)})
    selected_parents = {r["listing_uid"] for r in selected}
    check("T-D-REC-LISTINGS", "أهلية التعرف في الإعلان توافق الصور المؤهلة وتحفظ هوية الماركة والموديل",
          all((r["usable_recognition"] == "True") == (r["listing_uid"] in selected_parents) for r in listings)
          and all(r["rec_label"] == by_id[r["listing_uid"]]["make"] + "|" + by_id[r["listing_uid"]]["model"] for r in images))
    if "supported_make_labels" in summary:
        make_groups = defaultdict(lambda: defaultdict(set))
        for r in images:
            if r["image_uid"] in candidate_ids:
                make_groups[r["label_make"]][r["split"]].add(r["group_id"])
        supported_makes = {make for make, groups in make_groups.items()
                           if sum(len(v) for v in groups.values()) >= policy["min_total_groups"]
                           and all(len(groups[s]) >= policy["min_" + s + "_groups"] for s in ["train", "val", "test"])}
        selected_make_ids = {r["image_uid"] for r in images if r["usable_make_recognition"] == "True"}
        expected_make_ids = {r["image_uid"] for r in images if r["image_uid"] in candidate_ids and r["label_make"] in supported_makes}
        selected_make_parents = {r["listing_uid"] for r in images if r["image_uid"] in selected_make_ids}
        check("T-D-MAKE-SUPPORT", "أهلية تصنيف الماركة تحقق دعم الأقسام دون دمج موديلات نادرة",
              selected_make_ids == expected_make_ids
              and len(supported_makes) == summary["supported_make_labels"]
              and {r["make"] for r in read_csv(PROCESSED / "make_support.csv") if r["supported"] == "True"} == supported_makes
              and all((r["usable_make_recognition"] == "True") == (r["listing_uid"] in selected_make_parents) for r in listings)
              and all(r["label_make"] == by_id[r["listing_uid"]]["make"] and r["label_model"] == by_id[r["listing_uid"]]["model"] for r in images),
              {"makes": len(supported_makes), "images": len(selected_make_ids)})
    mileage_ok = True
    for r in listings:
        flags = set(r["data_quality_flags"].split(";"))
        expected = set()
        if r["condition"] == "used":
            km = float(r["km_mileage"])
            age = date.fromisoformat(r["date_collection"]).year - int(r["year"])
            if km == 100: expected.add("mileage_exact_100_review")
            if age <= 2 and km < 500: expected.add("young_used_low_mileage_review")
            if km / max(age, 1) > 60000: expected.add("annualized_mileage_proxy_high_review")
        relevant = flags & {"mileage_exact_100_review", "young_used_low_mileage_review", "annualized_mileage_proxy_high_review"}
        mileage_ok &= relevant == expected
    check("T-D-MILEAGE-REVIEW", "القيم غير المعتادة للممشى موسومة دون تعويض أو حذف", mileage_ok)
    used_prices = [float(r["sar_price"]) for r in listings if r["condition"] == "used"]
    q1, _, q3 = statistics.quantiles(used_prices, n=4, method="inclusive")
    lower, upper = q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)
    check("T-D-IQR-REVIEW", "علامات IQR وصفية صحيحة ولا تدخل سمات النموذج",
          all(("global_price_iqr_outlier_review" in r["data_quality_flags"].split(";")) ==
              (r["condition"] == "used" and not lower <= float(r["sar_price"]) <= upper) for r in listings)
          and "data_quality_flags" not in schema["price_features_allowlist"])
    price_rows = read_csv(PROCESSED / "price_dataset.csv")
    def eligible_price(r):
        if r["condition"] != "used": return False
        if not content_policy: return True
        unusual = set(r["data_quality_flags"].split(";")) & {"mileage_exact_100_review", "young_used_low_mileage_review", "annualized_mileage_proxy_high_review"}
        return not unusual or r["mileage_source_status"] == "source_confirms_record"
    check("T-D-PRICE-SCOPE", "مجموعة السعر تطابق المستعمل المؤهل دون الجديد والممشى غير المعتاد غير المؤكد",
          {r["listing_uid"] for r in price_rows} == {r["listing_uid"] for r in listings if eligible_price(r)}
          and all((r["usable_price"] == "True") == eligible_price(r) for r in listings))
    check("T-D-OPTIONAL-POLICY", "الحقول الاختيارية الناقصة لا تدخل ميزات السعر ولا يفترض ممشى الجديد صفرًا",
          not {"color", "body_type", "listing_date"} & set(schema["price_features_allowlist"])
          and all(r["km_mileage"] == "" and json.loads(r["field_states"])["km_mileage"] == "not_required_for_new"
                  for r in listings if r["condition"] == "new"))
    date_errors = []
    for row in listings:
        if row["listing_date"]:
            try:
                date.fromisoformat(row["listing_date"])
            except ValueError:
                date_errors.append(row["listing_uid"])
    check("T-D-LISTING-DATES", "تواريخ الإعلان المتوفرة صالحة بصيغة ISO، مع تمييزها عن تاريخ الجمع", not date_errors, date_errors[:10])
    if summary.get("enrichment", {}).get("complete"):
        from day20_enrichment_policy import inspect_saved_page
        from day20_enrich import read_events
        events = read_events()
        queue = {r["listing_uid"]: r for r in load_json(ROOT / "data/audit/day20/enrichment_queue.json")}
        all_rows_by_uid = {r["listing_uid"]: r for r in listings + quarantine}
        check("T-D-ENRICH-COMPLETE", "كل حالة في قائمة الاستكمال لها نتيجة مصدر موثقة",
              events.keys() == queue.keys()
              and len(events) == summary["enrichment"]["queue_size"]
              and all_rows_by_uid.keys() >= events.keys()
              and all(record["external_listing_id"] == queue[key]["external_listing_id"] == all_rows_by_uid[key]["external_listing_id"] for key, record in events.items()))
        paths = {r["parsed_path"] for r in events.values() if r["status"] == "ok"}
        for row in listings + quarantine:
            paths.update(p["path"] for p in json.loads(row["field_provenance"]).values()
                         if p.get("path", "").startswith("data/recovery/") and p["path"].endswith(".json"))
        paths.update(r["page_evidence"] for r in images if r.get("page_evidence", "").endswith(".json"))
        with ThreadPoolExecutor(max_workers=4) as pool:
            pages = dict(pool.map(inspect_saved_page, sorted(paths)))
        identity_ok = all(str(pages[record["parsed_path"]]["external_listing_id"]) == all_rows_by_uid[key]["external_listing_id"]
                          for key, record in events.items() if record["status"] == "ok")
        identity_ok &= all(str(pages[evidence["path"]]["external_listing_id"]) == row["external_listing_id"]
                           for row in listings + quarantine for evidence in json.loads(row["field_provenance"]).values()
                           if evidence.get("path") in pages)
        check("T-D-ENRICH-EVIDENCE", "كل دليل استكمال مطابق لهوية المصدر وHTML وبصمته ووسوم معرضه", len(pages) == len(paths) and identity_ok, {"verified_pages": len(pages)})
        from urllib.parse import urlsplit
        gallery_errors = []
        for image in images:
            if not image["image_kind"]:
                continue
            page = pages.get(image["page_evidence"])
            filename = urlsplit(image["source_url"]).path.rsplit("/", 1)[-1]
            if not page or str(page["external_listing_id"]) != by_id[image["listing_uid"]]["external_listing_id"] or not any(
                filename in {urlsplit(photo["url"]).path.rsplit("/", 1)[-1], urlsplit(photo.get("gallery_url", photo["url"])).path.rsplit("/", 1)[-1]}
                and photo["image_kind"] == image["image_kind"] for photo in page["photos"]
            ):
                gallery_errors.append(image["image_uid"])
        check("T-D-GALLERY-LABELS", "وسم خارجية أو داخلية الصورة مطابق لمعرض إعلانها نفسه", not gallery_errors, gallery_errors[:10])
        labels_ok = True
        mileage_ok = True
        for row in listings + quarantine:
            provenance = json.loads(row["field_provenance"])
            for field in ["color", "body_type", "listing_date"]:
                if row[field]:
                    page = pages.get(provenance[field]["path"])
                    value = str(page["fields"].get(field) or "") if page else ""
                    if field in ["color", "body_type"]: value = value.lower()
                    if field == "color": value = value.replace("grey", "gray")
                    labels_ok &= row[field] == value
            evidence = provenance.get("current_source_mileage", {})
            page = pages.get(evidence.get("path"))
            expected_current = number(page["fields"].get("km_mileage")) if page else None
            expected_status = "not_required_for_new" if row["condition"] == "new" and row["km_mileage"] == "" else (
                "source_unavailable" if not page else "source_not_provided" if expected_current is None else
                "source_confirms_record" if expected_current == number(row["km_mileage"]) else "source_differs_historical_record")
            mileage_ok &= row["mileage_source_status"] == expected_status
            mileage_ok &= number(row["current_source_mileage"]) == (None if expected_status == "not_required_for_new" else expected_current)
        check("T-D-ENRICH-LABELS", "اللون والهيكل والتاريخ مأخوذة من حقول المصدر نفسها دون تخمين", labels_ok)
        check("T-D-MILEAGE-SOURCE", "حالة الممشى تفصل تأكيد المصدر واختلاف اللقطات وتعذر الاسترجاع", mileage_ok)
    if content_policy:
        from day20_content_validate import content_checks
        for result in content_checks(PROCESSED, listings, images, schema, summary):
            check(*result)
    if check_raw:
        manifest = load_json(AUDIT / "raw_manifest.json")
        def inspect(record):
            errors = []
            for key in ["original_path", "raw_path"]:
                path = ROOT / record[key]
                if not path.is_file() or sha256(path) != record["sha256"]:
                    errors.append(record[key])
            return errors
        errors = []
        with ThreadPoolExecutor(max_workers=4) as pool:
            for result in pool.map(inspect, manifest["files"]):
                errors.extend(result)
        original_count = sum(sum(p.is_file() for p in (ROOT / source).rglob("*")) for source in SOURCES)
        raw_count = sum(sum(p.is_file() for p in (RAW / source).rglob("*")) for source in SOURCES)
        check("T-D-RAW", "الأصول والنسخة الخام لم تتغير بعد المعالجة", not errors and original_count == raw_count == manifest["total_files"], {"files": manifest["total_files"], "errors": errors[:10]})
    result = {"checked_at": now(), "processed_data_path": relative(PROCESSED), "passed": all(r["passed"] for r in checks), "passed_count": sum(r["passed"] for r in checks), "total_checks": len(checks), "checks": checks}
    save_json(AUDIT / "validation.json", result)
    summary["validation"] = {k: result[k] for k in ["checked_at", "passed", "passed_count", "total_checks"]}
    summary["day20_status"] = "complete_with_documented_quarantine" if result["passed"] and check_raw else "validation_incomplete_or_failed"
    save_json(PROCESSED / "quality_report.json", summary)
    if content_policy and result["passed"] and check_raw:
        save_json(AUDIT / "content_validation_manifest.json", {p.relative_to(PROCESSED).as_posix(): sha256(p) for p in PROCESSED.rglob("*") if p.is_file()})
    return result

if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-dir", type=Path, default=PROCESSED)
    args = parser.parse_args()
    PROCESSED = args.processed_dir.resolve()
    raise SystemExit(0 if validate()["passed"] else 1)
