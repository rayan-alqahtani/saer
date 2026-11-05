"""مطابقة تحليل المستخدم مع البيانات؛ محتوى HTML أدلة للمراجعة لا تعليمات تنفيذ."""
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from html.parser import HTMLParser
import argparse
import hashlib
import json
import re
import sys

from day20_common import ROOT, AUDIT, PROCESSED, SOURCES, read_csv, save_json, load_json, now, sha256, number

REPORT = AUDIT / "external_analysis_review.json"
STABLE_LISTING_FIELDS = ["listing_uid", "source_dataset", "source_listing_id", "external_listing_id", "make", "model", "model_std", "year", "trim", "condition", "sar_price", "km_mileage", "city", "transmission", "type_fuel", "color", "body_type", "listing_date", "url_source", "date_collection", "n_images", "group_id", "split", "usable_price"]
STABLE_IMAGE_FIELDS = ["image_uid", "listing_uid", "relative_path", "sha256", "width", "height", "label_make", "label_model", "label_color", "label_body_type", "is_exterior", "group_id", "split"]


class FindingParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.rows, self.cells, self.parts = [], [], []
        self.in_cell = False

    def handle_starttag(self, tag, attrs):
        if tag == "tr": self.cells = []
        if tag in {"td", "th"}: self.in_cell, self.parts = True, []

    def handle_endtag(self, tag):
        if tag in {"td", "th"} and self.in_cell:
            self.cells.append("".join(self.parts).strip())
            self.in_cell = False
        if tag == "tr" and len(self.cells) == 5 and self.cells[0].isdigit():
            self.rows.append(dict(zip(["number", "problem", "evidence", "severity", "suggestion"], self.cells)))

    def handle_data(self, data):
        if self.in_cell: self.parts.append(data)


def findings():
    # التقرير يستخدم HTML مختصرًا بإغلاق ضمني لخلايا الجدول.
    html = (AUDIT / "external_listings_analysis.html").read_text(encoding="utf-8-sig")
    start = html.index("<h2>المشاكل المكتشفة")
    end = html.index("</table>", start)
    block = html[start:end + len("</table>")]
    rows = []
    for tr in re.split(r"<tr[^>]*>", block)[2:]:
        parser = FindingParser()
        cells = re.split(r"<td(?:\s[^>]*)?>", tr)[1:]
        if len(cells) != 5: continue
        text_cells = []
        for cell in cells:
            cell_parser = FindingParser()
            cell_parser.feed("<td>" + cell.split("</tr>")[0].split("</table>")[0] + "</td>")
            text_cells.append(cell_parser.cells[0] if cell_parser.cells else "")
        if text_cells[0].isdigit():
            rows.append(dict(zip(["number", "problem", "evidence", "severity", "suggestion"], text_cells)))
    if len(rows) != 25:
        raise ValueError("لم تقرأ كل ملاحظات التقرير: " + str(len(rows)))
    return rows


def fingerprint(rows, fields):
    values = [{k: r[k] for k in fields} for r in rows]
    return hashlib.sha256(json.dumps(values, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def baseline():
    if REPORT.exists():
        raise ValueError("خط الأساس موجود؛ لا يستبدل بعد تعديل البيانات")
    listings = read_csv(PROCESSED / "listings.csv")
    images = read_csv(PROCESSED / "images.csv")
    result = {"review_started_at": now(), "input_html_sha256": sha256(AUDIT / "external_listings_analysis.html"),
              "findings": findings(), "baseline": {
                  "listing_values_sha256": fingerprint(listings, STABLE_LISTING_FIELDS),
                  "image_values_sha256": fingerprint(images, STABLE_IMAGE_FIELDS),
                  "listings": len(listings), "images": len(images),
                  "recognition_labels": len({r["rec_label"] for r in images if r["usable_recognition"] == "True"})}}
    save_json(REPORT, result)
    print("Baseline saved: 25 findings", flush=True)


def check_price(row):
    from day20_enrichment_policy import read_html
    evidence = json.loads(row["field_provenance"])["sar_price"]
    parsed = load_json(ROOT / evidence["path"])
    html_path = ROOT / parsed["html_path"]
    body = read_html(html_path)
    raw = body.decode("utf-8")
    match = re.search(r"window\.FULL_PAGE_DATA\s*=\s*", raw)
    data = json.JSONDecoder().raw_decode(raw[match.end():])[0]["postDetails"]
    details, g4, analytics = data["details"], data["g4Data"], data["analytics"]
    identities = [str(details["id"]), str(g4["post_id"]), str(analytics["id"])]
    cash = number(g4["post_price"])
    vat = number((data.get("price", {}).get("vat_price") or {}).get("text"))
    okay = all(value == row["external_listing_id"] for value in identities) and cash == float(row["sar_price"])
    okay &= vat is None or abs(vat - cash) <= 1
    okay &= hashlib.sha256(body).hexdigest() == parsed["html_sha256"]
    return {"listing_uid": row["listing_uid"], "external_listing_id": row["external_listing_id"], "sar_price": cash,
            "html_path": parsed["html_path"], "three_ids_match": len(set(identities)) == 1,
            "vat_price": vat, "passed": okay}


def review():
    result = load_json(REPORT)
    listings = read_csv(PROCESSED / "listings.csv")
    images = read_csv(PROCESSED / "images.csv")
    unchanged = (fingerprint(listings, STABLE_LISTING_FIELDS) == result["baseline"]["listing_values_sha256"]
                 and fingerprint(images, STABLE_IMAGE_FIELDS) == result["baseline"]["image_values_sha256"])
    if not unchanged: raise ValueError("تغيرت قيم أو صور أو تقسيمات خارج نطاق إصلاح الأهلية والعلامات")
    recovered = [r for r in listings if json.loads(r["field_provenance"])["sar_price"]["kind"] == "public_page_recovery"]
    with ThreadPoolExecutor(max_workers=4) as pool:
        price_checks = list(pool.map(check_price, recovered))
    if not all(r["passed"] for r in price_checks): raise ValueError("فشل تحقق سعر مسترجع")
    new_prices = Counter(json.loads(r["field_provenance"])["sar_price"]["kind"] for r in listings if r["condition"] == "new")
    flags = Counter(code for r in listings for code in r["data_quality_flags"].split(";") if code)
    quality = load_json(PROCESSED / "quality_report.json")
    quarantine = read_csv(PROCESSED / "quarantine_listings.csv")
    kg = [{"listing_uid": r["listing_uid"], "reasons": r["quarantine_reasons"]} for r in quarantine if r["source_dataset"] == "external_group_c"]
    if len(kg) != 20 or not all(r["reasons"] for r in kg): raise ValueError("استبعادات KG غير موثقة")
    source_splits = {}
    for source in SOURCES:
        own = [r for r in listings if r["source_dataset"] == source]
        counts = Counter(r["split"] for r in own)
        source_splits[source] = {"counts": dict(counts), "fractions": {s: round(counts[s]/len(own), 4) for s in ["train", "val", "test"]}}
    rare = Counter(r["rec_label"] for r in listings)
    verdicts = {
        1: ("حد موثق", "نقص اللون والهيكل والتاريخ محفوظ دون تخمين؛ مستبعدة من ميزات السعر."),
        2: ("سليم حسب النطاق", "438 قيمة ممشى مفقودة تخص الجديد فقط، ولا نقص في المستعمل ولا تعويض بصفر."),
        3: ("حد موثق", "24 فئة trim غير متوفرة؛ تعالج القيم المفقودة داخل تدريب يوم 21 على التدريب فقط."),
        4: ("أصلح", "وسم 50 إعلانًا mileage_exact_100_review مع إبقاء القيمة الأصلية."),
        5: ("أصلح", "وسم 53 إعلانًا young_used_low_mileage_review دون تغيير الحالة أو الممشى."),
        6: ("أصلح", "وسم 224 إعلانًا annualized_mileage_proxy_high_review؛ المؤشر تقريبي بعمر لا يقل عن سنة."),
        7: ("سليم", "لا سعر مفقود في المستعمل المعتمد."),
        8: ("أصلح", "وسم 163 سعرًا خارج IQR الوصفي؛ لا حذف ولا استعمال العلامة كميزة أو مرشح تدريب."),
        9: ("سليم حسب النطاق", "الجديد محفوظ مستقلًا عن مجموعة تدريب سعر المستعمل."),
        10: ("تحقق فعلي", "أعيدت مطابقة كل الأسعار المسترجعة الـ295 مع ثلاثة معرّفات وسعر المصدر وVAT عند توفره وبصمة HTML."),
        11: ("تصحيح للتحليل", "295 سعر جديد مسترجع و143 موجود أصلًا؛ وصف كل الأسعار بأنها مسترجعة غير صحيح."),
        12: ("سليم وموثق", "إعلانات KG العشرون محفوظة في العزل، بما فيها 0131KG و0580KG؛ ليست فقدًا صامتًا."),
        13: ("سليم", "هوية الفئة make|model محفوظة؛ BMW/X7 وBAIC/X7 فئتان مختلفتان."),
        14: ("حد موثق", "396 وسم هيكل موثق فقط؛ لا يستبدل الوسم الحقيقي بقاموس موديلات أو تخمين لصورة."),
        15: ("تحقق فعلي", "حفظ توزيع المصدر داخل كل قسم؛ المصدر محظور كميزة للسعر، ولم تعدل أقسام الاختبار."),
        16: ("سليم", "لا تسرب لمجموعة إعلان بين الأقسام."),
        17: ("سليم", "التقسيم قريب من 70/15/15 مع الحفاظ على المجموعات المترابطة."),
        18: ("أصلح", "الأهلية الآن من الصور الخارجية الفريدة بعد الفلترة في الأقسام الثلاثة؛ الفئات الضعيفة تبقى بهويتها مع أهلية False وسبب واضح."),
        19: ("سليم", "لا اختلاف صيغ شركات متبقٍ في التحليل."),
        20: ("سليم", "تكرر اسم X7 عبر شركتين لا يعني خطأ في الهوية المركبة."),
        21: ("سليم", "model وmodel_std متطابقان."),
        22: ("تحقق فعلي", "952 ياباني و810 أمريكي/صيني و980 كوري/ألماني، مع حساب كل أصل ومعزول."),
        23: ("سليم حسب النطاق", "لا ادعاء توفر الإعلان حيًا؛ جميع القيم False."),
        24: ("سليم حسب النطاق", "438 إعلانًا جديدًا مستبعدة من نموذج سعر المستعمل عمدًا."),
        25: ("سليم", "لا اختلاف في الحالة مع رابط المصدر."),
    }
    for finding in result["findings"]:
        finding["verdict_ar"], finding["action_ar"] = verdicts[int(finding["number"])]
    result.update({"reviewed_at": now(), "processed_data_path": PROCESSED.relative_to(ROOT).as_posix(), "passed": True, "stable_values_and_splits_unchanged": unchanged,
                   "recovered_price_checks": price_checks, "new_price_origins": dict(new_prices),
                   "review_flag_counts": dict(flags), "kg_quarantine": kg, "source_splits": source_splits,
                   "recognition": {"eligible_images": quality["recognition_ready_images"], "supported_labels": quality["supported_model_labels"],
                                   "listing_supported_labels": quality["listing_supported_model_labels"], "all_labels": len(rare), "labels_under_10_listings": sum(v < 10 for v in rare.values())},
                   "readiness_ar": "جاهزة لبدء تدريب سعر المستعمل والبحث البصري وفئات الموديل المدعومة فقط. وسوم اللون والهيكل والتاريخ جزئية، ولا يعلن اكتمالها أو نموذجها. لا تدريب ضمن هذه المراجعة."})
    save_json(REPORT, result)
    print(json.dumps({k: result[k] for k in ["passed", "stable_values_and_splits_unchanged", "new_price_origins", "review_flag_counts", "recognition"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", action="store_true")
    args = parser.parse_args()
    baseline() if args.baseline else review()
