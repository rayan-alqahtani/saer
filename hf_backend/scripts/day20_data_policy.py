"""سياسات مراجعة القيم وأهلية الصور؛ لا تعويض أو حذف تخميني."""
from collections import defaultdict
from datetime import date
import math

RECOGNITION_POLICY = {"min_total_groups": 12, "min_train_groups": 8,
                      "min_val_groups": 1, "min_test_groups": 1}


def yes(value):
    return value is True or str(value).lower() == "true"


def reviewed_overview(image):
    """وسم معرض المصدر لا يكفي: يفشل مغلقًا دون مراجعة المحتوى."""
    return image.get("content_review_status") == "reviewed" and image.get("content_kind") == "vehicle_overview"


def overview_exclusion_status(image):
    return "content_review_pending" if image.get("content_review_status") != "reviewed" else "reviewed_not_vehicle_overview"


def mileage_flags(row):
    if row.get("condition") != "used" or row.get("km_mileage") in (None, ""):
        return []
    mileage = float(row["km_mileage"])
    # مؤشر سنوي تقريبي مثبت إلى سنة الجمع؛ لا تاريخ استخدام أول موثوق هنا.
    age = date.fromisoformat(row["date_collection"]).year - int(row["year"])
    flags = []
    if mileage == 100:
        flags.append("mileage_exact_100_review")
    if age <= 2 and mileage < 500:
        flags.append("young_used_low_mileage_review")
    if mileage / max(age, 1) > 60000:
        flags.append("annualized_mileage_proxy_high_review")
    return flags


def linear_quantile(values, fraction):
    ordered = sorted(values)
    point = (len(ordered) - 1) * fraction
    lower = math.floor(point)
    return ordered[lower] + (point - lower) * (ordered[math.ceil(point)] - ordered[lower])


def descriptive_price_bounds(rows):
    values = [float(r["sar_price"]) for r in rows if r["condition"] == "used"]
    if len(values) < 4:
        return None
    q1, q3 = linear_quantile(values, .25), linear_quantile(values, .75)
    return q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)


def assign_recognition(listings, images):
    """الأهلية تعتمد مناظر السيارة المراجعة والفريدة ضمن الأقسام الثابتة."""
    by_id = {r["listing_uid"]: r for r in listings}
    listing_groups = defaultdict(lambda: defaultdict(set))
    photo_groups = defaultdict(lambda: defaultdict(set))
    candidates = []
    seen = set()
    for row in listings:
        row["rec_label"] = row["make"] + "|" + row["model"]
        listing_groups[row["rec_label"]][row["split"]].add(row["group_id"])
    for image in sorted(images, key=lambda r: r["image_uid"]):
        row = by_id[image["listing_uid"]]
        image["split"] = row["split"]
        image["rec_label"] = row["rec_label"]
        image["usable_visual_search"] = False
        image["usable_recognition"] = False
        image["exterior_status"] = "source_gallery_kind_missing" if image["is_exterior"] == "" else (
            "source_gallery_exterior" if yes(image["is_exterior"]) else "source_gallery_interior")
        if not reviewed_overview(image):
            image["recognition_status"] = overview_exclusion_status(image)
        elif image["sha256"] in seen:
            image["recognition_status"] = "duplicate_exact_image"
        else:
            seen.add(image["sha256"])
            image["usable_visual_search"] = True
            candidates.append(image)
            photo_groups[row["rec_label"]][row["split"]].add(row["group_id"])
    support_rows = []
    supported = set()
    reasons = {}
    for label, counts in sorted(listing_groups.items()):
        photos = photo_groups[label]
        sizes = {s: len(photos[s]) for s in ["train", "val", "test"]}
        failures = []
        if sum(sizes.values()) < RECOGNITION_POLICY["min_total_groups"]:
            failures.append("insufficient_exterior_groups")
        for split in ["train", "val", "test"]:
            if sizes[split] < RECOGNITION_POLICY["min_" + split + "_groups"]:
                failures.append("insufficient_" + split + "_groups")
        reasons[label] = ";".join(failures) or "supported"
        if not failures:
            supported.add(label)
        listing_supported = sum(len(v) for v in counts.values()) >= 12 and len(counts["train"]) >= 8 and bool(counts["val"]) and bool(counts["test"])
        support_rows.append({"label": label, **{s + "_groups": len(counts[s]) for s in ["train", "val", "test"]},
                             "listing_supported": listing_supported,
                             **{"exterior_" + s + "_groups": sizes[s] for s in ["train", "val", "test"]},
                             "supported": not failures, "recognition_status": reasons[label]})
    eligible_listings = set()
    for image in candidates:
        image["recognition_status"] = reasons[image["rec_label"]]
        if image["rec_label"] in supported:
            image["usable_recognition"] = True
            eligible_listings.add(image["listing_uid"])
    for row in listings:
        row["usable_recognition"] = row["listing_uid"] in eligible_listings
        row["recognition_status"] = "supported" if row["usable_recognition"] else (
            "no_eligible_exterior_image" if row["rec_label"] in supported else reasons[row["rec_label"]])
    return support_rows, supported


def assign_make_recognition(listings, images):
    """تجهيز تصنيف الماركة أولًا، مع بقاء هوية الموديل مستقلة حتى للفئات النادرة."""
    groups = defaultdict(lambda: defaultdict(set))
    by_id = {r["listing_uid"]: r for r in listings}
    candidates, seen = [], set()
    for image in sorted(images, key=lambda r: r["image_uid"]):
        image["usable_make_recognition"] = False
        if not reviewed_overview(image):
            image["make_recognition_status"] = overview_exclusion_status(image)
        elif image["sha256"] in seen:
            image["make_recognition_status"] = "duplicate_exact_image"
        else:
            seen.add(image["sha256"])
            candidates.append(image)
            groups[image["label_make"]][image["split"]].add(image["group_id"])
    supported, support_rows = set(), []
    for make in sorted({r["make"] for r in listings}):
        sizes = {s: len(groups[make][s]) for s in ["train", "val", "test"]}
        okay = sum(sizes.values()) >= RECOGNITION_POLICY["min_total_groups"] and all(
            sizes[s] >= RECOGNITION_POLICY["min_" + s + "_groups"] for s in sizes)
        if okay: supported.add(make)
        support_rows.append({"make": make, **{s + "_exterior_groups": sizes[s] for s in sizes}, "supported": okay})
    eligible_parents = set()
    for image in candidates:
        image["usable_make_recognition"] = image["label_make"] in supported
        image["make_recognition_status"] = "supported" if image["usable_make_recognition"] else "insufficient_brand_groups"
        if image["usable_make_recognition"]: eligible_parents.add(image["listing_uid"])
    for row in listings:
        row["usable_make_recognition"] = row["listing_uid"] in eligible_parents
    return support_rows, supported
