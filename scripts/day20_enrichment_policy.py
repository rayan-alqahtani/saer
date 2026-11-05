"""اعتماد أدلة الاستكمال المكتمل؛ يحفظ اللقطة التاريخية والسجلات المرفوضة."""
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
from day20_common import ROOT, RECOVERY, load_json, number


def read_html(path):
    path = Path(path)
    data = path.read_bytes()
    return gzip.decompress(data) if path.suffix == ".gz" else data


def verified_evidence(parsed):
    if not parsed.get("page_identity_verified"):
        return False
    path = (ROOT / parsed["html_path"]).resolve()
    if not path.is_relative_to(RECOVERY) or not path.is_file():
        return False
    return hashlib.sha256(read_html(path)).hexdigest() == parsed["html_sha256"]


def inspect_saved_page(relative_path):
    from day20_common import public_page_data
    parsed = load_json(ROOT / relative_path)
    if not verified_evidence(parsed):
        raise ValueError("دليل المصدر أو بصمته غير مطابق: " + relative_path)
    actual = public_page_data(read_html(ROOT / parsed["html_path"]).decode("utf-8", "replace"), parsed["external_listing_id"])
    if actual["fields"] != parsed["fields"]:
        raise ValueError("الحقول المستخرجة تختلف عن HTML: " + relative_path)
    expected_photos = [(p["url"], p["image_kind"]) for p in actual["photos"]]
    if expected_photos != [(p["url"], p["image_kind"]) for p in parsed["photos"]]:
        raise ValueError("وسوم المعرض تختلف عن HTML: " + relative_path)
    return relative_path, parsed


def overlay_completed(recovery):
    summary = load_json(RECOVERY / "enrichment/summary.json", {})
    if not summary:
        return recovery, {}
    if not summary.get("complete"):
        raise ValueError("استكمال المصدر جارٍ؛ لا تعتمد بيانات جزئية قبل انتهاء جميع الحالات")
    from day20_enrich import read_events
    events = read_events()
    if len(events) != summary["queue_size"]:
        raise ValueError("سجل الاستكمال لا يغطي قائمة الحالات")
    merged = {**recovery, "pages": dict(recovery["pages"])}
    merged["images"] = list(recovery["images"])
    for key, record in events.items():
        if record["status"] == "ok":
            parsed = load_json(ROOT / record["parsed_path"])
            if str(parsed.get("external_listing_id")) != record["external_listing_id"] or not verified_evidence(parsed):
                raise ValueError("دليل استكمال غير مطابق: " + key)
            merged["pages"][key] = {"status": "ok", "parsed_path": record["parsed_path"],
                                     "at": parsed["fetched_at"], "reasons": record["reasons"], "enrichment": True}
        elif key not in merged["pages"] or merged["pages"][key].get("status") != "ok":
            merged["pages"][key] = {**record, "enrichment": True}
    image_log = RECOVERY / "enrichment/image_events.jsonl"
    if image_log.exists():
        paths = {r["relative_path"] for r in merged["images"]}
        for line in image_log.read_text(encoding="utf-8").splitlines():
            item = json.loads(line)
            if item.get("status") == "ok" and item["relative_path"] not in paths:
                merged["images"].append(item)
                paths.add(item["relative_path"])
    return merged, summary


def mileage_source_status(condition, stored_value, page):
    if condition == "new" and stored_value in (None, ""):
        return "not_required_for_new", None
    if page is None:
        return "source_unavailable", None
    current = number(page["fields"].get("km_mileage"))
    if current is None:
        return "source_not_provided", None
    return ("source_confirms_record" if current == number(stored_value) else "source_differs_historical_record"), current
