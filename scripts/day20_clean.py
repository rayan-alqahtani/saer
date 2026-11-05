"""تنظيف المصادر مستقلة، ثم دمج مشروط ببوابات الجودة؛ لا تدريب هنا."""
from __future__ import annotations
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import argparse
import json
import math
import statistics
import sys
from day20_common import *
from day20_inventory import image_stats
from day20_quality import duplicate_groups
from day20_data_policy import assign_recognition, assign_make_recognition, descriptive_price_bounds, mileage_flags, RECOGNITION_POLICY
from day20_enrichment_policy import overlay_completed, mileage_source_status

ISSUE_FIELDS = ["issue_id", "source_dataset", "entity_type", "entity_uid", "field", "issue_code", "original_value", "final_value", "action", "evidence"]
CORE_FIELDS = ["make", "model", "year", "condition", "sar_price", "city", "transmission", "type_fuel"]

def truth(value):
    return value is True or str(value).lower() == "true"

def canonical_make(make):
    return {"Mercedes-Benz": "Mercedes", "Mercedes Benz": "Mercedes", "Mini": "MINI", "Lynk and Co": "Lynk & Co"}.get(text(make), text(make))

def field_state(raw, value, not_required=False):
    if value is None or value == "":
        if not_required:
            return "not_required_for_new"
        return "unknown" if text(raw).lower() in {"unknown", "other", "غير معروف"} else "missing" if not text(raw) else "invalid"
    return "known"

def split_groups(listings, seed=42):
    """تقسيم حتمي على مجموعات الإعلان، مع موازنة الماركة والموديل دون sklearn."""
    fractions = {"train": .7, "val": .15, "test": .15}
    groups = defaultdict(list)
    for row in listings:
        groups[row["group_id"]].append(row)
    strata = {g: Counter(r["make"] + "|" + r["model"] for r in rows) for g, rows in groups.items()}
    total = Counter()
    remaining = Counter()
    for counter in strata.values():
        total.update(counter)
        remaining.update(counter.keys())
    support = remaining.copy()
    assigned = {s: Counter() for s in fractions}
    global_n = Counter()
    ordered = sorted(groups, key=lambda g: (-len(groups[g]), hashlib.sha256(f"{seed}|{g}".encode()).hexdigest()))
    result = {}
    for gid in ordered:
        counter = strata[gid]
        allowed = set(fractions)
        for label in counter:
            if support[label] >= 3:
                missing = {s for s in fractions if assigned[s][label] == 0}
                if remaining[label] <= len(missing):
                    intersection = allowed & missing
                    if intersection:
                        allowed = intersection
            elif not any(assigned[s][label] for s in fractions):
                allowed &= {"train"}
        if not allowed:
            allowed = set(fractions)
        def cost(split):
            delta = 0.0
            for label, count in counter.items():
                target = total[label] * fractions[split]
                old = assigned[split][label]
                delta += ((old + count - target) ** 2 - (old - target) ** 2) / max(target, 1)
            target = len(listings) * fractions[split]
            old = global_n[split]
            delta += .2 * ((old + len(groups[gid]) - target) ** 2 - (old - target) ** 2) / max(target, 1)
            return delta, ["train", "val", "test"].index(split)
        choice = min(allowed, key=cost)
        result[gid] = choice
        assigned[choice].update(counter)
        global_n[choice] += len(groups[gid])
        remaining.subtract(counter.keys())
    return result

def clean(preview=False):
    if load_json(PROCESSED / "schema.json", {}).get("content_review_required"):
        raise SystemExit("هذه نسخة مراجعة بصريًا؛ لا تعاد تهيئتها بخط التنظيف التاريخي. استخدم Validate وTests وReport، وأعد تجهيز أي مراجعة جديدة في مسار مرحلي مستقل.")
    before = load_json(AUDIT / "before.json")
    recovery = load_json(RECOVERY / "manifest.json", {})
    if not before:
        raise SystemExit("لم يكتمل الجرد الأولي.")
    if not preview and (not recovery.get("complete") or not recovery.get("fallback_complete")):
        raise SystemExit("بوابة الدمج مغلقة: الاسترجاع لم ينته.")
    transient = load_json(RECOVERY / "transient_checks.json", {})
    if not preview and transient and not transient.get("complete"):
        raise SystemExit("بوابة الدمج مغلقة: تحقق أخطاء الخادم المؤقتة لم ينته.")
    if not preview and transient.get("complete"):
        known_paths = {r["relative_path"] for r in recovery["images"]}
        extras = [r for r in transient.get("images", []) if r["relative_path"] not in known_paths]
        if extras:
            recovery["images"].extend(extras)
        recovery["transient_verification"] = "data/recovery/transient_checks.json"
        save_json(RECOVERY / "manifest.json", recovery)
    recovery, enrichment_summary = overlay_completed(recovery)
    raw_integrity = {source: True for source in SOURCES}
    if not preview:
        manifest = load_json(AUDIT / "raw_manifest.json")
        def raw_check(entry):
            path = ROOT / entry["raw_path"]
            source = Path(entry["raw_path"]).parts[2]
            return source, path.is_file() and sha256(path) == entry["sha256"]
        with ThreadPoolExecutor(max_workers=4) as pool:
            for source, okay in pool.map(raw_check, manifest["files"]):
                raw_integrity[source] &= okay
        if not all(raw_integrity.values()):
            raise SystemExit("بوابة الدمج مغلقة: تغيرت النسخة الخام عن بصمتها.")
    issues = []
    changes = []
    listings = []
    pages = {}
    def issue(source, entity, entity_uid, field, code, original="", final="", action="documented", evidence=""):
        issues.append({"issue_id": f"D20-{len(issues)+1:06d}", "source_dataset": source, "entity_type": entity, "entity_uid": entity_uid, "field": field, "issue_code": code, "original_value": original, "final_value": final, "action": action, "evidence": evidence})
    for source, filename in SOURCES.items():
        raw_path = RAW / source / filename
        cached = cache_rows(source)
        for ordinal, raw in enumerate(read_csv(raw_path), 2):
            key = uid(source, raw["id_listing"])
            original_cache = cached.get(raw["external_listing_id"], {})
            historical_identity_bad = str(original_cache.get("post_id")) != raw["external_listing_id"] or original_cache.get("id_match") is False
            initial = effective_initial(raw, original_cache)
            model, model_origin = inferred_model(initial)
            values = {k: text(v) for k, v in initial.items()}
            values["make"] = canonical_make(values["make"])
            values["model"] = model
            values["condition"] = text(values.get("condition")).lower()
            provenance = {k: {"kind": "raw_csv", "path": relative(raw_path), "row": ordinal} for k in [*CORE_FIELDS, "trim", "km_mileage"]}
            for field in ["sar_price", "km_mileage", "trim", "city", "transmission", "type_fuel"]:
                if not text(raw.get(field)) and text(initial.get(field)):
                    provenance[field] = {"kind": "identity_verified_raw_cache", "path": relative(RAW / source / "raw_cache/pages.jsonl"), "external_listing_id": raw["external_listing_id"]}
            provenance["model"]["kind"] = model_origin
            provenance["model"]["source_fields"] = ["model", "model_raw", "trim"] if model_origin == "source_model_and_trim" else ["model", "model_raw"]
            provenance["model"]["normalization_rules"] = "scripts/day20_common.py:normalize_model"
            problems = []
            flags = []
            manifest_entry = recovery.get("pages", {}).get(key, {})
            recovered = None
            if manifest_entry.get("status") == "ok":
                recovered = load_json(ROOT / manifest_entry["parsed_path"])
                pages[key] = recovered
                fresh = recovered["fields"]
                if historical_identity_bad:
                    for field in [*CORE_FIELDS, "trim", "km_mileage"]:
                        values[field] = fresh.get(field, "")
                        provenance[field] = {"kind": "public_page_recovery_after_wrong_historical_id", "path": manifest_entry["parsed_path"], "fetched_at": recovered["fetched_at"]}
                    values["make"] = canonical_make(values["make"])
                    model = normalize_model(values["make"], values["model"], values.get("trim"))
                    values["model"] = model
                    model_origin = "public_page_recovery"
                    issue(source, "listing", key, "external_listing_id", "historical_identity_mismatch", original_cache.get("post_id"), raw["external_listing_id"], "fixed", manifest_entry["parsed_path"])
                for field in ["make", "year", "condition"]:
                    old = canonical_make(values.get(field)) if field == "make" else number(values.get(field)) if field == "year" else text(values.get(field)).lower()
                    new = canonical_make(fresh.get(field)) if field == "make" else number(fresh.get(field)) if field == "year" else text(fresh.get(field)).lower()
                    if old and new and old != new:
                        problems.append("source_identity_conflict_" + field)
                        issue(source, "listing", key, field, "source_identity_conflict", old, new, "quarantined", manifest_entry["parsed_path"])
                for field in ["sar_price", "km_mileage", "trim", "city", "transmission", "type_fuel"]:
                    if not text(values.get(field)) and text(fresh.get(field)):
                        values[field] = fresh[field]
                        provenance[field] = {"kind": "public_page_recovery", "path": manifest_entry["parsed_path"], "fetched_at": recovered["fetched_at"]}
                fresh_model = normalize_model(values["make"], fresh.get("model"), fresh.get("trim"))
                identity_conflict = any(code.startswith("source_identity_conflict_") for code in problems)
                exact_source_model = enrichment_summary.get("complete") and not identity_conflict and fresh_model and model_key(fresh_model) not in AMBIGUOUS_MODELS and model_key(fresh_model) != model_key(model)
                if (not model or model_origin == "listing_url_unverified" or model_key(model) in AMBIGUOUS_MODELS or exact_source_model) and fresh_model:
                    values["model"] = fresh_model
                    provenance["model"] = {"kind": "public_page_enrichment" if exact_source_model else "public_page_recovery", "path": manifest_entry["parsed_path"], "fetched_at": recovered["fetched_at"], "normalization_rules": "scripts/day20_common.py:normalize_model"}
                    model_origin = "public_page_recovery"
                elif fresh_model and model_key(model) != model_key(fresh_model):
                    flags.append("historical_model_differs_from_current_page")
                    issue(source, "listing", key, "model", "historical_model_difference", model, fresh_model, "documented", manifest_entry["parsed_path"])
            elif manifest_entry:
                status = manifest_entry.get("status")
                if status == "'postDetails'":
                    status = "source_redirected_or_unavailable"
                flags.append("recovery_" + str(status))
                issue(source, "listing", key, "source_page", "recovery_unavailable", "", status, "documented", manifest_entry.get("html_path", "data/recovery/network.jsonl"))
            if historical_identity_bad and not recovered:
                problems.append("historical_identity_mismatch")
                issue(source, "listing", key, "external_listing_id", "historical_identity_mismatch", original_cache.get("post_id"), raw["external_listing_id"], "quarantined", relative(RAW / source / "raw_cache/pages.jsonl"))
            for field in ["year", "sar_price", "km_mileage"]:
                values[field] = number(values.get(field))
            if values["year"] is not None and values["year"].is_integer():
                values["year"] = int(values["year"])
            else:
                problems.append("invalid_year")
            if not values["year"] or not 2018 <= values["year"] <= 2026:
                problems.append("year_out_of_scope")
            if values["sar_price"] is None or not 8000 <= values["sar_price"] <= 3_000_000:
                problems.append("missing_or_invalid_price")
            if values["condition"] not in {"used", "new"}:
                problems.append("invalid_condition")
            if values["condition"] == "used" and (values["km_mileage"] is None or not 0 <= values["km_mileage"] <= 600000):
                problems.append("missing_or_invalid_used_mileage")
            if not values["model"] or model_key(values["model"]) in AMBIGUOUS_MODELS:
                problems.append("ambiguous_or_missing_model")
            if model_origin == "listing_url_unverified":
                problems.append("unverified_model_from_url")
            for field in ["make", "city", "transmission", "type_fuel"]:
                if not text(values.get(field)):
                    problems.append("missing_" + field)
            if not raw["external_listing_id"].isdigit() or not re.search(r"-" + re.escape(raw["external_listing_id"]) + r"/?$", raw["url_source"]):
                problems.append("invalid_source_identity")
            values["color"] = text(recovered["fields"].get("color")).lower().replace("grey", "gray") if recovered else ""
            values["body_type"] = text(recovered["fields"].get("body_type")).lower() if recovered else ""
            values["listing_date"] = text(recovered["fields"].get("listing_date")) if recovered else ""
            for field in ["color", "body_type", "listing_date"]:
                if values[field]:
                    provenance[field] = {"kind": "public_page_recovery", "path": manifest_entry["parsed_path"], "fetched_at": recovered["fetched_at"]}
                    changes.append({"source_dataset": source, "listing_uid": key, "field": field, "before": "", "after": values[field], "evidence": json.dumps(provenance[field], ensure_ascii=False)})
                    issue(source, "listing", key, field, "recovered_optional_label", "", values[field], "fixed", manifest_entry["parsed_path"])
            field_states = {field: field_state(raw.get(field), values.get(field), field == "km_mileage" and values["condition"] == "new") for field in [*CORE_FIELDS, "trim", "km_mileage", "color", "body_type", "listing_date"]}
            for field in [*CORE_FIELDS, "trim", "km_mileage"]:
                old, new = text(raw.get(field)), text(values.get(field))
                equivalent = old == new or (number(old) is not None and number(old) == number(new))
                if not equivalent or str(raw.get(field, "")) != text(raw.get(field)):
                    item = {"source_dataset": source, "listing_uid": key, "field": field, "before": raw.get(field, ""), "after": values.get(field), "evidence": json.dumps(provenance.get(field, {}), ensure_ascii=False)}
                    changes.append(item)
                    issue(source, "listing", key, field, "recovered_field" if not old else "normalized_field", old, new, "fixed", json.dumps(provenance.get(field, {}), ensure_ascii=False))
                if not new:
                    core = field in CORE_FIELDS or field == "km_mileage" and values["condition"] == "used"
                    issue(source, "listing", key, field, "missing_required_value" if core else "missing_optional_value", old, "", "quarantined" if core else "documented", manifest_entry.get("parsed_path", relative(raw_path)))
            row = {
                "listing_uid": key, "id_listing": key, "source_dataset": source, "source_listing_id": raw["id_listing"], "external_listing_id": raw["external_listing_id"],
                **{k: values.get(k, "") for k in [*CORE_FIELDS, "trim", "km_mileage", "color", "body_type", "listing_date"]},
                "model_std": values["model"], "url_source": raw["url_source"], "date_collection": raw["date_collection"],
                "raw_metadata_path": relative(raw_path), "raw_row_number": ordinal,
                "status_permission": raw["status_permission"], "processed_at": now(),
                "field_states": json.dumps(field_states, ensure_ascii=False, sort_keys=True),
                "field_provenance": json.dumps(provenance, ensure_ascii=False, sort_keys=True),
                "data_quality_flags": flags, "quarantine_reasons": list(dict.fromkeys(problems)),
                "live_availability_verified": False,
            }
            row["mileage_source_status"], row["current_source_mileage"] = mileage_source_status(values["condition"], values["km_mileage"], recovered)
            row["source_verified_at"] = recovered["fetched_at"] if recovered else ""
            row["trim_status"] = "provided" if values.get("trim") else "source_not_provided" if recovered else "source_unavailable"
            if recovered:
                provenance["current_source_mileage"] = {"kind": "source_verification", "path": manifest_entry["parsed_path"], "fetched_at": recovered["fetched_at"]}
                row["field_provenance"] = json.dumps(provenance, ensure_ascii=False, sort_keys=True)
            for code in mileage_flags(row):
                row["data_quality_flags"].append(code)
                issue(source, "listing", key, "km_mileage", code, row["km_mileage"], row["km_mileage"], "documented", row["raw_metadata_path"])
            listings.append(row)
    # علامات وصفية فقط؛ لا تستخدم إحصاءات الاختبار لتغيير الهدف أو حذف الأسعار.
    price_groups = defaultdict(list)
    for row in listings:
        if row["sar_price"] and row["sar_price"] > 0:
            price_groups[(row["make"], row["model"], row["condition"])].append(row)
    outlier_rows = []
    for group in price_groups.values():
        if len(group) < 8:
            continue
        logs = [math.log(r["sar_price"]) for r in group]
        center = statistics.median(logs)
        mad = statistics.median(abs(x-center) for x in logs)
        for row, value in zip(group, logs):
            if abs(value-center) > max(math.log(2.5), 5 * 1.4826 * mad):
                row["data_quality_flags"].append("descriptive_price_outlier")
                outlier_rows.append({"listing_uid": row["listing_uid"], "make": row["make"], "model": row["model"], "condition": row["condition"], "sar_price": row["sar_price"], "group_size": len(group), "log_deviation": round(abs(value-center), 5), "action": "retained_as_advertised_no_imputation"})
                issue(row["source_dataset"], "listing", row["listing_uid"], "sar_price", "descriptive_price_outlier", row["sar_price"], row["sar_price"], "documented", row["raw_metadata_path"])
    save_csv(AUDIT / "price_outlier_flags.csv", outlier_rows, ["listing_uid", "make", "model", "condition", "sar_price", "group_size", "log_deviation", "action"])
    by_uid = {r["listing_uid"]: r for r in listings}
    print("اكتمل تنظيف الحقول مستقلًا؛ بدء فحص صور الاسترجاع والتكرارات.", flush=True)
    images = read_csv(AUDIT / "images_before.csv")
    for row in images:
        row["is_valid"] = truth(row["is_valid"])
        row["low_resolution"] = truth(row["low_resolution"])
        row["width"], row["height"] = int(row["width"] or 0), int(row["height"] or 0)
        row["image_origin"] = "raw"
        row["source_url"] = ""
        row["page_evidence"] = ""
        row["image_kind"] = ""
    recovery_images = recovery.get("images", [])
    for recovered in recovery_images:
        recovered_path = ROOT / recovered["relative_path"]
        if not recovered_path.is_file() or sha256(recovered_path) != recovered["sha256"]:
            raise SystemExit("بوابة الدمج مغلقة: صورة الاسترجاع لا تطابق دليلها: " + recovered["relative_path"])
    stat_cache_path = AUDIT / "recovered_image_stats.csv"
    stat_cache = {r["relative_path"]: r for r in read_csv(stat_cache_path)} if stat_cache_path.exists() else {}
    jobs = [r for r in recovery_images if r["relative_path"] not in stat_cache]
    with ThreadPoolExecutor(max_workers=4) as pool:
        new_stats = list(pool.map(image_stats, [ROOT / r["relative_path"] for r in jobs]))
    for row in new_stats:
        stat_cache[row["relative_path"]] = row
    save_csv(stat_cache_path, sorted(stat_cache.values(), key=lambda r: r["relative_path"]))
    for recovered in recovery_images:
        row = dict(stat_cache[recovered["relative_path"]])
        row["is_valid"] = truth(row["is_valid"])
        row["low_resolution"] = truth(row["low_resolution"])
        row["width"], row["height"] = int(row["width"]), int(row["height"])
        row.update(source_dataset=recovered["source_dataset"], source_listing_id=recovered["id_listing"], listing_uid=recovered["listing_uid"], image_uid=recovered["listing_uid"] + ":" + Path(row["relative_path"]).stem, image_origin="recovered", source_url=recovered["url"], page_evidence=recovered["page_evidence"], image_kind=recovered.get("image_kind", ""))
        images.append(row)
        issue(row["source_dataset"], "image", row["image_uid"], "file", "recovered_image", "", row["relative_path"], "fixed", row["page_evidence"])
    for source in SOURCES:
        original_meta = {r["path_image"].replace("\\", "/"): r for r in read_csv(RAW / source / "car_images_downloaded.csv")}
        for row in images:
            if row["source_dataset"] != source or row["image_origin"] != "raw":
                continue
            local = Path(row["relative_path"]).relative_to(Path("data/raw") / source).as_posix()
            meta = original_meta.get(local, {})
            row["source_url"] = meta.get("url_image", "")
            page = pages.get(row["listing_uid"])
            if page:
                name = urllib_basename(row["source_url"])
                matching = [p for p in page["photos"] if urllib_basename(p["url"]) == name]
                if matching:
                    row["image_kind"] = matching[0]["image_kind"]
                    row["page_evidence"] = recovery["pages"][row["listing_uid"]]["parsed_path"]
    image_groups, listing_groups, pairs, structured_pairs, conflicts = duplicate_groups(images, listings)
    duplicate_summary = Counter(p["kind"] for p in pairs)
    print(json.dumps({"image_duplicate_pairs": dict(duplicate_summary), "structured_listing_pairs": len(structured_pairs), "conflicting_images": len(conflicts)}, ensure_ascii=False), flush=True)
    image_exclusions = []
    clean_images = []
    image_counts = Counter()
    seen_same_listing = set()
    for image in sorted(images, key=lambda r: (r["image_origin"] != "raw", r["image_uid"])):
        source, key = image["source_dataset"], image["listing_uid"]
        reasons = []
        if key not in by_uid:
            reasons.append("orphan_image")
        if not image["is_valid"]:
            reasons.append("invalid_image")
        if image["low_resolution"]:
            reasons.append("low_resolution")
        if number(image.get("pixel_std")) is not None and number(image["pixel_std"]) < 3:
            reasons.append("near_blank_image")
        if image["image_uid"] in conflicts:
            reasons.append("visual_label_conflict")
        hash_key = (key, image["sha256"])
        if hash_key in seen_same_listing:
            reasons.append("duplicate_same_listing")
        if reasons:
            image_exclusions.append({**image, "exclusion_reasons": ";".join(reasons)})
            issue(source, "image", image["image_uid"], "file", ";".join(reasons), image["relative_path"], "", "quarantined", "data/audit/day20/duplicate_pairs.csv")
            continue
        seen_same_listing.add(hash_key)
        listing = by_uid[key]
        image_counts[key] += 1
        image.update(
            id_listing=key, path_image=image["relative_path"], extension=Path(image["relative_path"]).suffix,
            label_make=listing["make"], label_model=listing["model"], label_color=listing["color"], label_body_type=listing["body_type"],
            duplicate_group=image_groups.find(image["image_uid"]), group_id=listing_groups.find(key),
            is_exterior=True if image["image_kind"] == "exterior" else False if image["image_kind"] == "interior" else "",
        )
        clean_images.append(image)
    eligible_used = [r for r in listings if not r["quarantine_reasons"] and image_counts[r["listing_uid"]] and r["condition"] == "used"]
    price_bounds = descriptive_price_bounds(eligible_used)
    iqr_rows = []
    if price_bounds:
        for row in eligible_used:
            if not price_bounds[0] <= row["sar_price"] <= price_bounds[1]:
                row["data_quality_flags"].append("global_price_iqr_outlier_review")
                iqr_rows.append({"listing_uid": row["listing_uid"], "sar_price": row["sar_price"], "lower_bound": price_bounds[0], "upper_bound": price_bounds[1], "action": "retained_descriptive_only_not_model_feature"})
                issue(row["source_dataset"], "listing", row["listing_uid"], "sar_price", "global_price_iqr_outlier_review", row["sar_price"], row["sar_price"], "documented", row["raw_metadata_path"])
    save_csv(AUDIT / "global_price_iqr_flags.csv", iqr_rows, ["listing_uid", "sar_price", "lower_bound", "upper_bound", "action"])
    for row in listings:
        row["n_images"] = image_counts[row["listing_uid"]]
        row["group_id"] = listing_groups.find(row["listing_uid"])
        if not row["n_images"]:
            row["quarantine_reasons"].append("no_verified_usable_image")
            issue(row["source_dataset"], "listing", row["listing_uid"], "images", "no_verified_usable_image", "", 0, "quarantined", recovery.get("pages", {}).get(row["listing_uid"], {}).get("html_path", row["raw_metadata_path"]))
        for code in row["quarantine_reasons"]:
            if code in {"ambiguous_or_missing_model", "unverified_model_from_url", "year_out_of_scope", "invalid_condition", "invalid_source_identity", "invalid_year"}:
                issue(row["source_dataset"], "listing", row["listing_uid"], "model" if "model" in code else "record", code, "", "", "quarantined", row["raw_metadata_path"])
        row["quarantine_reasons"] = ";".join(sorted(set(row["quarantine_reasons"])))
        row["data_quality_flags"] = ";".join(sorted(set(row["data_quality_flags"])))
        row["usable_price"] = not row["quarantine_reasons"] and row["condition"] == "used"
        row["usable_recognition"] = False  # تحسم بعد فحص دعم الصور الخارجية الفعلي.
    # كل رابط مفقود في الكتالوج الأصلي له مصير؛ لا يمر كرابط معطوب إلى المخرجات.
    for source in SOURCES:
        for missing in read_csv(AUDIT / source / "missing_image_slots_before.csv"):
            key = uid(source, missing["id_listing"])
            match = [r for r in recovery_images if r["listing_uid"] == key and str(r.get("original_slot_no")) == str(missing["image_no"])]
            issue(source, "historical_image_slot", key + ":slot" + missing["image_no"], "url_image", "historical_missing_image", missing["url_image"], match[0]["relative_path"] if match else "", "fixed" if match else "quarantined", match[0]["page_evidence"] if match else "data/recovery/manifest.json")
    for pair in pairs:
        issue(pair["left_listing_uid"].split(":", 1)[0], "image_pair", pair["left_image_uid"] + "|" + pair["right_image_uid"], "duplicate_group", pair["kind"], "", "same_split_group", "fixed", "data/audit/day20/duplicate_pairs.csv")
    for pair in structured_pairs:
        issue(pair["left_listing_uid"].split(":", 1)[0], "listing_pair", pair["left_listing_uid"] + "|" + pair["right_listing_uid"], "group_id", pair["kind"], "", "same_split_group", "fixed", "data/audit/day20/structured_duplicates.csv")
    source_gates = {}
    clean_listings = [r for r in listings if not r["quarantine_reasons"]]
    clean_ids = {r["listing_uid"] for r in clean_listings}
    final_images = [r for r in clean_images if r["listing_uid"] in clean_ids]
    excluded_parent = [r for r in clean_images if r["listing_uid"] not in clean_ids]
    image_exclusions.extend({**r, "exclusion_reasons": "parent_listing_quarantined"} for r in excluded_parent)
    for source in SOURCES:
        own = [r for r in listings if r["source_dataset"] == source]
        accepted = [r for r in clean_listings if r["source_dataset"] == source]
        excluded = [r for r in own if r["quarantine_reasons"]]
        own_images = [r for r in final_images if r["source_dataset"] == source]
        source_issues = [r for r in issues if r["source_dataset"] == source]
        checks = {
            "unique_listing_uids": len({r["listing_uid"] for r in own}) == len(own),
            "all_original_rows_accounted_for": len(accepted) + len(excluded) == before["sources"][source]["listings"],
            "no_required_missing_in_accepted": all(all(text(r.get(k)) for k in CORE_FIELDS) for r in accepted),
            "every_accepted_listing_has_valid_image": all(r["n_images"] > 0 for r in accepted),
            "no_ambiguous_models_in_accepted": all(model_key(r["model"]) not in AMBIGUOUS_MODELS for r in accepted),
            "all_image_parents_exist": all(r["listing_uid"] in clean_ids for r in own_images),
            "all_accepted_images_valid": all(r["is_valid"] for r in own_images),
            "all_issues_dispositioned": all(r["action"] in {"fixed", "quarantined", "documented"} for r in source_issues),
            "excel_csv_match": before["sources"][source]["excel_value_differences"] == 0,
            "raw_snapshot_unchanged": raw_integrity[source] if not preview else None,
        }
        gate = {"source_dataset": source, "checked_at": now(), "passed": all(checks.values()), "preview_only": preview, "checks": checks, "original_listings": len(own), "accepted_listings": len(accepted), "quarantined_listings": len(excluded), "accepted_images": len(own_images), "issue_actions": dict(Counter(r["action"] for r in source_issues)), "quarantine_reasons": dict(Counter(code for r in excluded for code in r["quarantine_reasons"].split(";")))}
        source_gates[source] = gate
        target = STAGING / source
        save_csv(target / "listings_all.csv", own)
        save_csv(target / "listings_clean.csv", accepted, list(listings[0]))
        save_csv(target / "listings_quarantine.csv", excluded, list(listings[0]))
        save_csv(target / "images_clean.csv", own_images, list(final_images[0]) if final_images else [])
        save_csv(target / "issues.csv", source_issues, ISSUE_FIELDS)
        save_json(target / "gate.json", gate)
        print(json.dumps(gate, ensure_ascii=False), flush=True)
    save_csv(AUDIT / "issues.csv", issues, ISSUE_FIELDS)
    save_csv(AUDIT / "changes.csv", changes, ["source_dataset", "listing_uid", "field", "before", "after", "evidence"])
    save_csv(AUDIT / "duplicate_pairs.csv", pairs, ["left_image_uid", "right_image_uid", "left_listing_uid", "right_listing_uid", "kind", "pixel_mae", "same_make"])
    save_csv(AUDIT / "structured_duplicates.csv", structured_pairs, ["left_listing_uid", "right_listing_uid", "kind"])
    save_csv(AUDIT / "image_exclusions.csv", image_exclusions)
    save_json(AUDIT / "source_gates.json", source_gates)
    if preview:
        print("معاينة فقط: لا جداول موحدة ولا دمج.", flush=True)
        return
    if not all(g["passed"] for g in source_gates.values()):
        raise SystemExit("بوابة الدمج مغلقة: فشل تحقق مصدر واحد أو أكثر.")
    print("اجتازت المصادر الثلاثة بواباتها؛ بدء الدمج الأولي الآن.", flush=True)
    split_map = split_groups(clean_listings)
    for row in clean_listings:
        row["split"] = split_map[row["group_id"]]
    support_rows, supported_labels = assign_recognition(clean_listings, final_images)
    make_support, supported_makes = assign_make_recognition(clean_listings, final_images)
    final_images.sort(key=lambda r: r["image_uid"])
    clean_listings.sort(key=lambda r: r["listing_uid"])
    save_csv(PROCESSED / "listings.csv", clean_listings)
    save_csv(PROCESSED / "images.csv", final_images)
    save_csv(PROCESSED / "quarantine_listings.csv", [r for r in listings if r["quarantine_reasons"]], list(listings[0]))
    save_csv(PROCESSED / "image_exclusions.csv", image_exclusions)
    save_csv(PROCESSED / "price_dataset.csv", [r for r in clean_listings if r["usable_price"]], list(clean_listings[0]))
    save_csv(PROCESSED / "recognition_images.csv", [r for r in final_images if r["usable_recognition"]], list(final_images[0]))
    save_csv(PROCESSED / "recognition_candidates.csv", final_images)
    save_csv(PROCESSED / "class_support.csv", support_rows)
    save_csv(PROCESSED / "make_support.csv", make_support)
    vocabulary = []
    for field in ["make", "model", "city", "transmission", "type_fuel", "condition", "color", "body_type"]:
        for value, count in sorted(Counter(r[field] for r in clean_listings).items()):
            vocabulary.append({"field": field, "value": value, "listing_count": count, "state": "known" if value else "missing_optional"})
    save_csv(PROCESSED / "categorical_vocabulary.csv", vocabulary)
    for split in ["train", "val", "test"]:
        save_csv(PROCESSED / "splits" / (split + ".csv"), [{k: r[k] for k in ["listing_uid", "group_id", "source_dataset", "make", "model"]} for r in clean_listings if r["split"] == split])
    split_metadata = {"seed": 42, "method": "deterministic_greedy_stratified_listing_duplicate_groups", "target_fractions": {"train": .7, "val": .15, "test": .15}, "actual_listings": dict(Counter(r["split"] for r in clean_listings)), "actual_groups": dict(Counter(split_map.values())), "duplicate_checks": {"exact_sha256": True, "perceptual_candidates": "phash<=3,dhash<=4,pixel_MAE<=6; conservatively grouped", "structured_duplicates": True}, "test_usage": "لم تستخدم بيانات الاختبار لتدريب أو ضبط نموذج؛ الفهارس فقط أعدت اليوم."}
    save_json(PROCESSED / "splits/metadata.json", split_metadata)
    schema = {"version": "day20-1", "created_at": now(), "language": "ar", "price_target": "سعر الإعلان بالريال السعودي، وليس سعر البيع النهائي", "price_scope": "used_only_until_new_model_evaluated", "price_features_allowlist": ["make", "model_std", "year", "km_mileage", "trim", "city", "transmission", "type_fuel"], "forbidden_model_features": ["listing_uid", "id_listing", "source_listing_id", "external_listing_id", "relative_path", "path_image", "url_source", "group_id", "split", "raw_metadata_path", "field_provenance"], "missing_values": {"missing": "لم يتوفر في المصدر", "unknown": "ذكر المصدر أنه غير معروف", "invalid": "قيمة غير صالحة", "not_required_for_new": "الممشى ليس شرطًا للجديد ولم يفترض صفرًا"}, "image_labels": "وسوم اللون والهيكل مخصبة فقط من الصفحة المطابقة، ووسم الخارج/الداخل من معرض المصدر؛ الصور غير الموسومة تنتظر فلترة يوم 21.", "raw_policy": "الأصول والنسخة الخام لا تعدل؛ الاسترجاع مستقل وله تاريخ ودليل.", "all_listing_columns": list(clean_listings[0]), "all_image_columns": list(final_images[0])}
    schema["version"] = "day20-2-reviewed"
    if enrichment_summary.get("complete"):
        schema["version"] = "day20-3-source-enriched"
        schema["source_enrichment_policy"] = "قيم سعر الإعلان وممشاه التاريخية محفوظة؛ current_source_mileage وsource_verified_at للمقارنة المنفصلة فقط. كل وسم مستكمل من صفحة الإعلان المطابقة، والفئات التي لم يذكرها المصدر لا تعوض بمستوى التجهيز."
    schema["recognition_policy"] = RECOGNITION_POLICY
    schema["forbidden_model_features"].extend(["source_dataset", "date_collection", "data_quality_flags", "recognition_status", "usable_price", "usable_recognition", "n_images"])
    schema["forbidden_model_features"].extend(["current_source_mileage", "source_verified_at", "mileage_source_status", "trim_status"])
    schema["forbidden_model_features"].append("usable_make_recognition")
    schema["descriptive_flags_policy"] = "علامات الممشى وIQR للمراجعة فقط؛ لا تغير القيم أو التقسيم ولا تدخل سمات النموذج. مؤشر الممشى السنوي تقريبي بسنة الجمع وحد أدنى سنة واحدة."
    schema["optional_labels_policy"] = "اللون والهيكل والتاريخ غير المتوفرة تبقى مفقودة؛ لا قاموس موديلات لإسناد وسم صورة غير مثبت."
    save_json(PROCESSED / "schema.json", schema)
    summary = {"created_at": now(), "day20_status": "awaiting_independent_validation", "source_gates_passed": True, "raw_listings": len(listings), "clean_listings": len(clean_listings), "quarantined_listings": len(listings) - len(clean_listings), "raw_images": before["raw_images"], "recovered_images": len(recovery_images), "clean_images": len(final_images), "excluded_images": len(image_exclusions), "price_rows": sum(r["usable_price"] for r in clean_listings), "verified_exterior_images": sum(r["is_exterior"] is True for r in final_images), "recognition_ready_images": sum(r["usable_recognition"] for r in final_images), "supported_model_labels": len(supported_labels), "known_color_listings": sum(bool(r["color"]) for r in clean_listings), "known_body_type_listings": sum(bool(r["body_type"]) for r in clean_listings), "remaining_required_missing_in_clean": 0, "issue_actions": dict(Counter(r["action"] for r in issues)), "duplicate_pairs": dict(duplicate_summary), "split": split_metadata}
    summary["listing_supported_model_labels"] = sum(r["listing_supported"] for r in support_rows)
    summary["recognition_policy"] = RECOGNITION_POLICY
    summary["review_flags"] = dict(Counter(flag for r in clean_listings for flag in r["data_quality_flags"].split(";") if flag))
    summary["enrichment"] = enrichment_summary
    summary["mileage_source_statuses"] = dict(Counter(r["mileage_source_status"] for r in clean_listings if any(flag in r["data_quality_flags"] for flag in ["mileage_exact_100_review", "young_used_low_mileage_review", "annualized_mileage_proxy_high_review"])))
    summary["supported_make_labels"] = len(supported_makes)
    summary["make_recognition_ready_images"] = sum(r["usable_make_recognition"] for r in final_images)
    listing_dates = [row["listing_date"] for row in clean_listings if row["listing_date"]]
    summary["listing_date_range"] = {"min": min(listing_dates), "max": max(listing_dates)} if listing_dates else None
    save_json(PROCESSED / "quality_report.json", summary)
    print(json.dumps(summary, ensure_ascii=False), flush=True)

def urllib_basename(url):
    from urllib.parse import urlsplit
    return urlsplit(url).path.rsplit("/", 1)[-1]

if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--preview", action="store_true", help="فحص مرحلي مستقل دون إنشاء جداول موحدة")
    args = parser.parse_args()
    clean(preview=args.preview)
