"""أدوات يوم 20 أكتوبر: أصول ثابتة، كتابة مع نسخ سابقة، ومعرّفات موثقة."""
from __future__ import annotations
import csv
import hashlib
import io
import json
import math
import re
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw"
AUDIT = ROOT / "data/audit/day20"
RECOVERY = ROOT / "data/recovery"
STAGING = ROOT / "data/staging/day20"
PROCESSED = ROOT / "data/processed"
SOURCES = {
    "external_group_a": "member1_japanese.csv",
    "external_group_b": "member2_american.csv",
    "external_group_c": "member3_korean_german.csv",
}
AR_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")

def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")

def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def relative(path):
    return Path(path).resolve().relative_to(ROOT).as_posix()

def save_bytes(path, content):
    """لا تضيع النتيجة السابقة عند إعادة التشغيل."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() == content:
            return
        tag = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
        old = ROOT / "data/history" / path.relative_to(ROOT)
        old = old.with_name(old.name + "." + tag)
        old.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, old)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_bytes(content)
    # قارئ أو مضاد فيروسات Windows قد يمسك الملف لحظيًا ويمنع الاستبدال الذري.
    for attempt in range(10):
        try:
            temp.replace(path)
            break
        except PermissionError:
            if attempt == 9:
                raise
            time.sleep(min(.05 * 2 ** attempt, 1.0))

def save_json(path, data):
    save_bytes(path, (json.dumps(data, ensure_ascii=False, indent=2, default=str) + "\n").encode("utf-8"))

def load_json(path, default=None):
    return json.loads(Path(path).read_text(encoding="utf-8-sig")) if Path(path).exists() else default

def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))

def save_csv(path, rows, fields=None):
    rows = list(rows)
    if fields is None:
        fields = list(dict.fromkeys(k for row in rows for k in row))
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields, extrasaction="raise", lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    save_bytes(path, buffer.getvalue().encode("utf-8-sig"))

def text(value):
    if value is None:
        return ""
    s = re.sub(r"\s+", " ", str(value).translate(AR_DIGITS).strip())
    return "" if s.lower() in {"", "nan", "none", "null", "n/a", "na"} else s

def number(value):
    s = text(value).replace(",", "").replace("٬", "").replace("٫", ".")
    if not s:
        return None
    try:
        x = float(s)
        return x if math.isfinite(x) else None
    except ValueError:
        return None

def uid(source, listing_id):
    return source + ":" + text(listing_id)

def cache_rows(source):
    path = RAW / source / "raw_cache/pages.jsonl"
    result = {}
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        if line.strip():
            row = json.loads(line)
            result[str(row.get("external_listing_id"))] = row
    return result

def model_key(value):
    return re.sub(r"[^a-z0-9]", "", text(value).lower())

# تهجئة وتنسيق فقط. لا ندمج موديلين مختلفين (مثل Optima وK5).
MODEL_ALIASES = {
    ("Toyota", "landcruiser70"): "Land Cruiser 70", ("Nissan", "xtrail"): "X-Trail",
    ("Mazda", "cx5"): "CX-5", ("Mazda", "cx9"): "CX-9",
    ("Mercedes", "sclass"): "S-Class",
    ("Chery", "arrizo6pro"): "Arrizo 6 PRO", ("Chery", "tiggo7promax"): "Tiggo 7 PRO MAX",
    ("Geely", "starray"): "Starray", ("BAIC", "u5plus"): "U5 Plus", ("Changan", "unis"): "UNI S",
    ("Toyota", "haice"): "Hiace", ("Mazda", "6"): "Mazda 6", ("Mazda", "3"): "Mazda 3",
    ("Mazda", "cx60"): "CX-60", ("Mazda", "cx90"): "CX-90",
    ("Mercedes", "c"): "C-Class", ("Mercedes", "e"): "E-Class", ("Mercedes", "s"): "S-Class",
    ("Mercedes", "a"): "A-Class", ("Mercedes", "g"): "G-Class",
    ("Hyundai", "santafe"): "Santa Fe", ("Volkswagen", "troc"): "T-Roc",
    ("Mitsubishi", "newoutlanderseries"): "Outlander",
}
AMBIGUOUS_MODELS = {"xseries", "unknown", "other"}

def normalize_model(make, value, trim=""):
    value = text(value)
    if make == "BMW" and model_key(value) == "xseries":
        # بعض صفحات المصدر تقسم «X2» إلى عائلة «X Series» وفئة «2 xDrive ...».
        # نقبل رقمًا صريحًا في بداية الفئة فقط، ولا نخمن من حجم المحرك أو الصورة.
        match = re.match(r"^X\s*([1-7])(?=\s|[sx]drive|m\d|$)", text(trim), re.I)
        if not match:
            match = re.match(r"^([1-7])\s+(?=[sx]drive\b|m\d)", text(trim), re.I)
        if match:
            return "X" + match.group(1)
    return MODEL_ALIASES.get((make, model_key(value)), value)

def inferred_model(row):
    """مخطط بديل من الحقل الأصلي؛ ثم رابط الإعلان نفسه عند الحاجة مع تعليم مصدره."""
    make = text(row.get("make"))
    raw = text(row.get("model")) or text(row.get("model_raw"))
    if raw:
        model = normalize_model(make, raw, row.get("trim"))
        return model, "source_model_and_trim" if make == "BMW" and model_key(raw) == "xseries" and model_key(model) != "xseries" else "original_field"
    match = re.search(r"/cardetail/(.+)-(?:new|used)-\d+/?$", text(row.get("url_source")))
    if not match:
        return "", "missing"
    slug = match.group(1)
    prefix = re.sub(r"[^a-z0-9]+", "-", make.lower()).strip("-")
    for candidate in [prefix, "mercedes-benz" if make == "Mercedes" else prefix]:
        if slug.startswith(candidate + "-"):
            slug = slug[len(candidate) + 1:]
            break
    else:
        return "", "missing"
    return normalize_model(make, " ".join(part.upper() if any(c.isdigit() for c in part) or len(part) <= 2 else part.title() for part in slug.split("-"))), "listing_url_unverified"

def effective_initial(row, cache):
    out = dict(row)
    # لا نستورد من سجل مخبأ لإعلان آخر.
    if str(cache.get("post_id")) == str(row.get("external_listing_id")) and cache.get("id_match") is not False:
        mapping = {"sar_price": "post_price", "km_mileage": "post_mileage", "trim": "post_ext", "city": "post_city", "transmission": "post_transmission", "type_fuel": "post_fuel", "model_raw": "post_model"}
        for field, key in mapping.items():
            if not text(out.get(field)) and text(cache.get(key)):
                out[field] = cache[key]
    return out

def recovery_reasons(row, downloaded_ids, expected_missing_ids):
    model, origin = inferred_model(row)
    reasons = []
    if row["id_listing"] not in downloaded_ids:
        reasons.append("no_downloaded_image")
    if row["id_listing"] in expected_missing_ids:
        reasons.append("failed_image_slot")
    for field in ["sar_price", "city", "transmission", "type_fuel", "trim"]:
        if not text(row.get(field)):
            reasons.append("missing_" + field)
    if not model or model_key(model) in AMBIGUOUS_MODELS or origin == "listing_url_unverified":
        reasons.append("unverified_model")
    if row.get("condition") == "used" and number(row.get("km_mileage")) is None:
        reasons.append("missing_used_mileage")
    return reasons

def public_page_data(html, sid):
    """نقرأ كائن الإعلان المحدد، لا أول post_id ولا الإعلانات المشابهة."""
    match = re.search(r"window\.FULL_PAGE_DATA\s*=\s*", html)
    if not match:
        raise ValueError("missing_full_page_data")
    obj = json.JSONDecoder().raw_decode(html[match.end():])[0]
    if "postDetails" not in obj:
        raise ValueError("source_redirected_or_unavailable")
    post = obj["postDetails"]
    details = post["details"]
    analytics = post.get("analytics") or {}
    g4 = post.get("g4Data") or {}
    if str(details.get("id")) != str(sid) or str(g4.get("post_id")) != str(sid) or str(analytics.get("id")) != str(sid):
        raise ValueError("identity_mismatch")
    if details.get("is_deleted"):
        raise ValueError("source_deleted")
    fields = {
        "make": text(g4.get("post_make")), "model": text(g4.get("post_model")),
        "year": number(g4.get("post_year")), "trim": text(g4.get("post_ext")),
        "condition": text(g4.get("post_condition")).lower(), "city": text(g4.get("post_city")),
        "transmission": text(g4.get("post_transmission")), "type_fuel": text(g4.get("post_fuel")),
        "sar_price": number(g4.get("post_price")),
        # g4Data هو المصدر الصريح؛ analytics يضع صفرًا افتراضيًا للجديد فلا ننقله.
        "km_mileage": number(g4.get("post_mileage")),
        "color": text(g4.get("post_exterior_color")), "body_type": text(analytics.get("shape")),
        "listing_date": text(g4.get("list_date")),
    }
    vat = number((post.get("price", {}).get("vat_price") or {}).get("text"))
    if vat is not None and fields["sar_price"] is not None and abs(vat - fields["sar_price"]) > 1:
        raise ValueError("cash_price_mismatch")
    photos = []
    for item in (post.get("gallery") or {}).get("images", []):
        url = text(item.get("img_url"))
        if re.match(r"https://cdn\.syarah\.com/", url) and f"/posts/{sid}/" in url:
            # حجم قريب من الصور الخام. لا نفترض امتداد الصورة من اسم URL.
            gallery_url = url
            url = re.sub(r"/0x\d+/", "/0x300/", url)
            photos.append({"url": url, "gallery_url": gallery_url, "image_kind": text(item.get("img_type")), "source_image_id": str(item.get("id", ""))})
    return {"fields": fields, "photos": photos, "source_sold_flag": details.get("is_sold"), "page_identity_verified": True}
