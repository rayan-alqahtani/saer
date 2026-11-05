"""تدقيق التغييرات منذ النسخة السابقة؛ لا يعتمد تغيير قيمة دون أصل قابل للتحقق."""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import argparse
import json
import sys
from pathlib import Path
from day20_common import ROOT, RAW, AUDIT, PROCESSED, SOURCES, read_csv, load_json, save_json, now, number, text, normalize_model, cache_rows, uid
from day20_clean import canonical_make
from day20_enrich import read_events

FIELDS = ["make", "model", "year", "condition", "sar_price", "km_mileage", "trim", "city", "transmission", "type_fuel", "color", "body_type", "listing_date"]
NUMERIC = {"year", "sar_price", "km_mileage"}


def equivalent(field, left, right):
    return number(left) == number(right) if field in NUMERIC else text(left) == text(right)


def original_snapshot_check():
    counts, mismatches, excluded_identity = Counter(), [], []
    for source, filename in SOURCES.items():
        cache = cache_rows(source)
        for row in read_csv(RAW / source / filename):
            saved = cache.get(row["external_listing_id"], {})
            if str(saved.get("post_id")) != row["external_listing_id"] or saved.get("id_match") is False:
                excluded_identity.append(uid(source, row["id_listing"]))
                continue
            for field, key in [("sar_price", "post_price"), ("km_mileage", "post_mileage")]:
                if number(row.get(field)) is not None and number(saved.get(key)) is not None:
                    counts[field] += 1
                    if number(row[field]) != number(saved[key]):
                        mismatches.append({"listing_uid": uid(source, row["id_listing"]), "field": field})
    if mismatches:
        raise ValueError("اختلاف في نقل قيمة من الكاش الأصلي المطابق للهوية")
    return {"compared_values": dict(counts), "mismatches": mismatches, "excluded_unmatched_cache_identities": excluded_identity,
            "scope_ar": "مطابقة النقل بين CSV وكاش لقطة المصدر الأصلية حيث تتوفر القيمتان وتتطابق الهوية؛ لا تدعي فحص العداد ميدانيًا أو تحقق كل سعر غير قابل للمقارنة."}


def source_value(row, field, evidence):
    path = ROOT / evidence["path"]
    if path.suffix == ".json":
        page = load_json(path)
        if str(page["external_listing_id"]) != row["external_listing_id"] or not page["page_identity_verified"]:
            raise ValueError("مرجع صفحة لا يخص الإعلان")
        values = page["fields"]
    elif path.suffix == ".csv":
        values = read_csv(path)[int(row["raw_row_number"]) - 2]
    elif path.suffix == ".jsonl":
        matches = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        matches = [item for item in matches if str(item.get("post_id")) == row["external_listing_id"] and item.get("id_match") is not False]
        if len(matches) != 1:
            raise ValueError("مرجع كاش غير محسوم الهوية")
        cached = matches[0]
        mapping = {"sar_price": "post_price", "km_mileage": "post_mileage", "trim": "post_ext", "city": "post_city", "transmission": "post_transmission", "type_fuel": "post_fuel", "model": "post_model"}
        values = {name: cached.get(key) for name, key in mapping.items()}
    else:
        raise ValueError("مرجع تغيير غير مدعوم: " + str(path))
    value = values.get(field)
    if field == "model": value = normalize_model(row["make"], value or values.get("model_raw"), values.get("trim"))
    elif field == "make": value = canonical_make(value)
    elif field in {"color", "body_type", "condition"}: value = text(value).lower()
    if field == "color": value = value.replace("grey", "gray")
    return value


def audit(processed, output):
    before = read_csv(AUDIT / "enrichment_baseline_listings.csv")
    before_q = read_csv(AUDIT / "enrichment_baseline_quarantine_listings.csv")
    accepted = read_csv(processed / "listings.csv")
    quarantine = read_csv(processed / "quarantine_listings.csv")
    old = {r["listing_uid"]: r for r in before + before_q}
    new = {r["listing_uid"]: r for r in accepted + quarantine}
    if old.keys() != new.keys(): raise ValueError("فقد أو زيادة إعلانات خارج المصادر الأصلية")
    old_accepted = {r["listing_uid"]: r for r in before}
    historical_values_preserved = all(
        equivalent(field, old_accepted[row["listing_uid"]][field], row[field])
        for row in accepted if row["listing_uid"] in old_accepted
        for field in ["sar_price", "km_mileage"]
    )
    if not historical_values_preserved:
        raise ValueError("تبدلت قيمة سعر أو ممشى تاريخية معتمدة أثناء استكمال الوسوم")
    changes = []
    counts = Counter()
    for key, row in new.items():
        provenance = json.loads(row["field_provenance"])
        for field in FIELDS:
            if equivalent(field, old[key].get(field), row.get(field)): continue
            evidence = provenance.get(field)
            if not evidence or not evidence.get("path"):
                raise ValueError("تغيير دون مصدر: " + key + ":" + field)
            value = source_value(row, field, evidence)
            if not equivalent(field, value, row[field]):
                raise ValueError("تغيير لا يطابق قيمته المصدرية: " + key + ":" + field)
            counts[field] += 1
            changes.append({"listing_uid": key, "field": field, "before": old[key].get(field), "after": row[field], "evidence": evidence["path"]})
    events = read_events()
    summary = load_json(ROOT / "data/recovery/enrichment/summary.json")
    if not summary["complete"] or len(events) != summary["queue_size"]:
        raise ValueError("قائمة المصدر لم تغلق")
    quality = load_json(processed / "quality_report.json")
    validation = load_json(output / "validation.json")
    if not validation["passed"] or validation["processed_data_path"] != processed.relative_to(ROOT).as_posix():
        raise ValueError("يلزم تحقق ناجح لنفس نسخة البيانات قبل إغلاق الملاحظات")
    from day20_external_review import check_price
    recovered_prices = [r for r in accepted if json.loads(r["field_provenance"])["sar_price"]["path"].startswith("data/recovery/")]
    with ThreadPoolExecutor(max_workers=4) as pool:
        price_checks = list(pool.map(check_price, recovered_prices))
    if not all(item["passed"] for item in price_checks):
        raise ValueError("فشل فحص مصدر سعر مسترجع")
    coverage = {f: {"before_known": sum(bool(r[f]) for r in before), "after_known": sum(bool(r[f]) for r in accepted),
                    "after_missing": sum(not r[f] for r in accepted)} for f in ["trim", "color", "body_type", "listing_date"]}
    old_ids = {r["listing_uid"] for r in before}
    new_ids = {r["listing_uid"] for r in accepted}
    result = {"checked_at": now(), "processed_data_path": processed.relative_to(ROOT).as_posix(), "passed": True,
              "all_original_rows_accounted_for": len(new), "all_changed_fields_source_verified": True,
              "historical_price_and_mileage_preserved": historical_values_preserved,
              "original_snapshot_check": original_snapshot_check(),
              "before_accepted": len(before), "after_accepted": len(accepted), "after_quarantine": len(quarantine),
              "rescued_listing_uids": sorted(new_ids - old_ids), "new_quarantine_uids": sorted(old_ids - new_ids),
              "field_change_counts": dict(counts), "source_verified_change_count": len(changes), "change_examples": changes[:100],
              "optional_coverage": coverage, "source_results": dict(Counter(r["status"] for r in events.values())),
              "listing_date_range": quality["listing_date_range"],
              "mileage_review_statuses": quality["mileage_source_statuses"],
              "recovered_price_checks": price_checks,
              "recognition": {k: quality[k] for k in ["supported_model_labels", "recognition_ready_images", "supported_make_labels", "make_recognition_ready_images", "verified_exterior_images"]},
              "scope_ar": "كل الحالات لها نتيجة موثقة: قيمة مصدرية مصادق عليها، قيمة تاريخية متميزة عن وقت التحقق، أو تعذر موثق للمصدر. لا تعويض للوسوم المجهولة ولا صور بديلة لإعلانات أخرى."}
    save_json(output / "enrichment_review.json", result)
    final_findings(accepted, quarantine, quality, result, output)
    print(json.dumps({k: result[k] for k in ["passed", "before_accepted", "after_accepted", "after_quarantine", "field_change_counts", "optional_coverage", "recognition"]}, ensure_ascii=False), flush=True)
    return result


def final_findings(rows, quarantine, quality, review, output):
    from day20_external_review import findings
    flags = Counter(code for row in rows for code in row["data_quality_flags"].split(";") if code)
    source_counts = Counter(row["source_dataset"] for row in rows)
    new_count = sum(row["condition"] == "new" for row in rows)
    origins = Counter("مسترجع" if json.loads(row["field_provenance"])["sar_price"]["path"].startswith("data/recovery/") else "أصلي" for row in rows if row["condition"] == "new")
    kg_quarantine = sum(row["source_dataset"] == "external_group_c" for row in quarantine)
    known = review["optional_coverage"]
    verdicts = {
        1: ("استكمل المتاح ووثق التعذر", f"اللون والهيكل والتاريخ متوفرة الآن لـ{known['color']['after_known']} إعلانًا؛ المتعذر {known['color']['after_missing']}. لكل قيمة دليل مطابق، ولا تعويض تخميني."),
        2: ("سليم حسب النطاق", f"{new_count} قيمة ممشى غير مشترطة تخص الجديد؛ لا ممشى مفقود في المستعمل المعتمد ولا تعويض بصفر."),
        3: ("حد المصدر موثق", f"بقيت {known['trim']['after_missing']} فئة trim غير متوفرة، مع حالة صريحة. لا يحل مستوى التجهيز محل اسم الفئة."),
        4: ("راجع المصدر ووسم", f"{flags['mileage_exact_100_review']} حالة ممشاها 100، مع حالة تأكيد المصدر أو تعذره. لم تختلق قيمة بديلة."),
        5: ("راجع المصدر ووسم", f"{flags['young_used_low_mileage_review']} حالة حديثة منخفضة الممشى؛ تبقى القيمة المعلنة مع دليل المراجعة."),
        6: ("راجع المصدر ووسم", f"{flags['annualized_mileage_proxy_high_review']} حالة بمؤشر سنوي مرتفع. العمر تقريبي بسنة الجمع وبمقام لا يقل عن سنة."),
        7: ("اجتاز الفحص", "لا سعر مفقود في المستعمل المعتمد."),
        8: ("عولج بعلامة وصفية", f"{flags['global_price_iqr_outlier_review']} سعرًا خارج IQR؛ لا حذف تلقائي ولا إدخال العلامة في ميزات النموذج."),
        9: ("نطاق منفصل", f"حفظت {new_count} إعلانًا جديدًا خارج مجموعة تدريب سعر المستعمل."),
        10: ("تحقق فعلي", f"طابقت {len(review['recovered_price_checks'])} سعرًا مسترجعًا مع ثلاثة معرّفات وسعر النقد والضريبة عند توفرها وبصمة HTML."),
        11: ("صحح وصف التحليل", f"مصادر أسعار الجديد في النسخة الحالية: {dict(origins)}. لا توصف الأسعار الأصلية بأنها مسترجعة."),
        12: ("كل أصل محسوب", f"الحالات الكورية والألمانية المعزولة الآن {kg_quarantine}، وكلها في quarantine_listings.csv بأسبابها؛ لا فقد صامت."),
        13: ("هوية مركبة محفوظة", "هوية التعرف make|model مستقلة لكل شركة وموديل؛ لا دمج عشوائي للفئات النادرة."),
        14: ("استكمل الوسم من المصدر", f"وسم الهيكل متوفر لـ{known['body_type']['after_known']} إعلانًا؛ المتعذر {known['body_type']['after_missing']}. لا قاموس موديلات بدل الوسم الحقيقي."),
        15: ("فحص التقسيم والمصدر", "أعيد التقسيم بعد التصحيحات مع تجميع التكرارات. المصدر محظور كميزة للسعر، وتوزيعه محفوظ لكل قسم."),
        16: ("اجتاز الفحص", "لا مجموعة إعلان أو صورة متطابقة أو مجموعة تكرار تعبر أقسام التدريب والتحقق والاختبار."),
        17: ("تقسيم موثق", f"الأعداد الفعلية بعد الحفاظ على مجموعات التكرار: {quality['split']['actual_listings']}، والهدف 70/15/15."),
        18: ("صححت أهلية الصور", f"{quality['recognition_ready_images']} صورة لموديلات مدعومة عددها {quality['supported_model_labels']}، و{quality['make_recognition_ready_images']} صورة لماركات عددها {quality['supported_make_labels']}. أهلية كل مهمة منفصلة بدعم الأقسام الفعلي."),
        19: ("تطبيع موثق", f"أسماء الماركات موحدة في القاموس وعددها {len({row['make'] for row in rows})}؛ أسماء الموديلات الأدق من الصفحة موثقة قبل وبعد."),
        20: ("سليم", "تكرر X7 تحت BMW وBAIC لا يخلط الفئتين، لأن مفتاح التعرف يتضمن الماركة."),
        21: ("اجتاز الفحص", "model وmodel_std متطابقان في كل إعلان معتمد."),
        22: ("كل المصادر محسوبة", f"التوزيع الفعلي للمعتمد: {dict(source_counts)}. المعتمد والمعزول يغطيان جميع السجلات الأصلية دون تداخل."),
        23: ("سليم حسب النطاق", "كل live_availability_verified=False. زيارة صفحة المصدر لا تثبت توفر السيارة للبيع الآن."),
        24: ("سليم حسب النطاق", f"مجموعة السعر تضم {quality['price_rows']} إعلانًا مستعملًا؛ الجديد محفوظ للاستخدام والتقييم المنفصل."),
        25: ("تحقق الهوية والحالة", "اختلاف حالة السجل عن الصفحة المطابقة يوجب العزل؛ لا تستنتج الحالة من غياب الممشى."),
    }
    if not all(row["model"] == row["model_std"] for row in rows):
        raise ValueError("اختلاف model وmodel_std")
    reviewed = findings()
    for finding in reviewed:
        finding["verdict_ar"], finding["action_ar"] = verdicts[int(finding["number"])]
    save_json(output / "external_analysis_final_review.json", {
        "reviewed_at": now(), "processed_data_path": review["processed_data_path"], "passed": True,
        "scope_ar": "نتيجة الملاحظات بعد الاستكمال الكامل. نجاح الفحص لا يعني اكتمال القيم التي لم يذكرها المصدر أو تعذر استرجاعه.",
        "findings": reviewed,
        "source_splits": {source: dict(Counter(row["split"] for row in rows if row["source_dataset"] == source)) for source in source_counts},
        "review_flag_counts": dict(flags), "new_price_origins": dict(origins),
    })


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-dir", type=Path, default=PROCESSED)
    parser.add_argument("--audit-dir", type=Path, default=AUDIT)
    args = parser.parse_args()
    audit(args.processed_dir, args.audit_dir)
