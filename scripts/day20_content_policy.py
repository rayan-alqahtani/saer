"""سياسات تجهيز المحتوى والوسوم فقط؛ لا نموذج ولا تعويض حقول المصدر."""
from collections import defaultdict
from day20_data_policy import RECOGNITION_POLICY, reviewed_overview, overview_exclusion_status

COLOR_FAMILIES = {
    "white": "white", "black": "black", "silver": "silver", "gray": "gray",
    "leaden": "gray", "dark leaden": "gray", "light gray": "gray", "nardo gray": "gray",
    "blue": "blue", "dark blue": "blue", "baby blue": "blue", "red": "red", "burgundy": "red",
    "green": "green", "light green": "green", "dark green": "green", "oil green": "green",
    "brown": "brown", "light brown": "brown", "dark brown": "brown", "beige": "beige",
    "golden": "gold", "light golden": "gold", "dark golden": "gold",
    "bronze": "bronze_copper", "dark yellow (copper)": "bronze_copper",
    "orange": "orange", "purple": "purple", "yellow": "yellow", "petrol": "", "": "",
}
BODY_FAMILIES = {"sedan": "sedan", "suv": "suv_crossover", "crossovers": "suv_crossover",
                 "hatchback": "hatchback", "pickup": "pickup", "van": "van", "sport": "", "": ""}
SPLITS = ("train", "val", "test")


def labels(row, visual=None):
    """القيم المشتقة مستقلة؛ unknown البصري يمنع الرجوع لتخمين المصدر."""
    color = COLOR_FAMILIES[row["color"]]
    body = BODY_FAMILIES[row["body_type"]]
    body_status = "source_shape_family" if body else "missing_or_ambiguous_source"
    if visual is not None:
        body = "" if visual["body_visual"] == "unknown" else visual["body_visual"]
        body_status = "visual_review" if body else "visual_insufficient_evidence"
    trim = " ".join(row["trim"].split())
    # لا دمج بين فئات تجهيز مختلفة؛ هذا اختلاف حالة أحرف موثق فقط.
    if row["make"] == "Nissan" and row["model_std"] == "Patrol" and trim.casefold() == "titanium":
        trim = "Titanium"
    return {"color_std": color, "color_label_status": "source_color_family" if color else (
                "ambiguous_source_color" if row["color"] else "missing_source_color"),
            "body_type_std": body, "body_label_status": body_status, "trim_std": trim}


def price_eligibility(row):
    if row["condition"] != "used":
        return False, "new_outside_price_scope"
    flags = set(row["data_quality_flags"].split(";"))
    unusual = flags & {"mileage_exact_100_review", "young_used_low_mileage_review", "annualized_mileage_proxy_high_review"}
    if unusual and row["mileage_source_status"] != "source_confirms_record":
        return False, "unusual_mileage_unconfirmed"
    return True, "eligible_historical_used_ad"


def support_tier(sizes):
    supported = sum(sizes.values()) >= RECOGNITION_POLICY["min_total_groups"] and all(
        sizes[s] >= RECOGNITION_POLICY["min_" + s + "_groups"] for s in SPLITS)
    # وصف حجم العينة وليس إثبات دقة؛ الحدود تثبت قبل تدريب أي نموذج.
    return "larger_evaluation_sample" if supported and sizes["train"] >= 20 and min(sizes["val"], sizes["test"]) >= 5 else (
        "experimental_minimum" if supported else "insufficient_support")


def assign_attribute_task(listings, images, task, label_field):
    """تعارض الوسم للصورة نفسها يعزل المهمة فقط، لا يخفي الصورة الأصلية."""
    groups = defaultdict(lambda: defaultdict(set))
    sha_labels = defaultdict(set)
    for im in images:
        if reviewed_overview(im) and im[label_field]:
            sha_labels[im["sha256"]].add(im[label_field])
    candidates, seen = [], set()
    for im in sorted(images, key=lambda x: x["image_uid"]):
        im["usable_" + task] = False
        if not reviewed_overview(im): status = overview_exclusion_status(im)
        elif not im[label_field]: status = "missing_or_ambiguous_label"
        elif len(sha_labels[im["sha256"]]) != 1: status = "conflicting_exact_image_labels"
        elif im["sha256"] in seen: status = "duplicate_exact_image"
        else:
            seen.add(im["sha256"])
            candidates.append(im)
            groups[im[label_field]][im["split"]].add(im["group_id"])
            status = "pending_support"
        im[task + "_status"] = status
    support = []
    tiers = {}
    for label in sorted({im[label_field] for im in images if im[label_field]}):
        sizes = {s: len(groups[label][s]) for s in SPLITS}
        tiers[label] = support_tier(sizes)
        support.append({"task": task, "label": label, **{s + "_groups": sizes[s] for s in SPLITS},
                        "supported": tiers[label] != "insufficient_support", "support_tier": tiers[label]})
    for im in candidates:
        im[task + "_status"] = tiers[im[label_field]]
        im["usable_" + task] = tiers[im[label_field]] != "insufficient_support"
    parents = {im["listing_uid"] for im in images if im["usable_" + task]}
    for row in listings:
        row["usable_" + task] = row["listing_uid"] in parents
    return support
