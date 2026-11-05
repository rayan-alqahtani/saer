"""تحقق مستقل من إضافة الصور؛ لا يستدعي دوال البناء أو اختيار المرشحين."""
from collections import Counter
from functools import lru_cache
import json
from urllib.parse import urlsplit
import numpy as np
from PIL import Image, ImageOps
from day20_common import *
from day20_enrichment_policy import read_html


def gallery_identity(photo, parent, parsed):
    url = urlsplit(photo["source_url"])
    if url.scheme != "https" or url.hostname != "cdn.example.invalid" or "/posts/" + parent["external_listing_id"] + "/" not in url.path:
        return False
    if str(parsed["external_listing_id"]) != parent["external_listing_id"] or not parsed.get("page_identity_verified"): return False
    actual = public_page_data(read_html(ROOT / parsed["html_path"]).decode("utf-8", "replace"), parent["external_listing_id"])
    return any(photo["source_url"] in {p["url"], p.get("gallery_url", p["url"])} and p["image_kind"] == photo["image_kind"] for p in actual["photos"])


def supplemental_checks(processed, listings, images, summary, manifest):
    checks = []
    def check(code, desc, okay, detail=""):
        checks.append(("T-S-" + code, desc, bool(okay), detail))
    base = AUDIT / "residual_completion"
    old_images = read_csv(base / "baseline_images.csv")
    old_listings = read_csv(base / "baseline_listings.csv")
    old_by_id = {r["image_uid"]: r for r in old_images}
    parents = {r["listing_uid"]: r for r in listings}
    added_ids = {r["image_uid"] for r in manifest["accepted_images"]}
    added = [r for r in images if r["image_uid"] in added_ids]
    rejected = manifest["rejected_images"]
    probe = load_json(base / "image_probe.json")
    log = [json.loads(line) for line in (base / "image_downloads.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    successes = [r for r in [probe, *log] if r.get("status") == "downloaded_pending_visual_review"]
    downloaded = {r["listing_uid"]: r for r in successes}
    matched_downloads = all(r["sha256"] == downloaded[r["listing_uid"]]["sha256"] and r["source_url"] == downloaded[r["listing_uid"]]["url"]
                            and r["page_evidence"] == downloaded[r["listing_uid"]]["page_evidence"] for r in added + rejected)
    preserved_fields = set(old_images[0]) - {"usable_recognition", "usable_make_recognition", "usable_visual_search", "usable_color", "usable_body_type",
                       "recognition_status", "make_recognition_status", "color_status", "body_type_status", "label_body_type_std", "body_label_status"}
    if summary.get("user_issues"): preserved_fields.discard("exterior_status")
    check("ACCOUNTING", "كل صورة مسترجعة محسوبة ومعرّفات الصور السابقة وبياناتها محفوظة",
          len(successes) == len(downloaded) == manifest["downloaded_images"] == len(added) + len(rejected)
          and {r["listing_uid"] for r in added + rejected} == set(downloaded)
          and set(old_by_id) | added_ids == {r["image_uid"] for r in images} and not set(old_by_id) & added_ids
          and matched_downloads and all(all(r[k] == old_by_id[r["image_uid"]][k] for k in preserved_fields) for r in images if r["image_uid"] in old_by_id),
          {"downloaded": len(successes), "accepted": len(added), "rejected": len(rejected)})
    identity_ok = all(gallery_identity(r, parents[r["listing_uid"]], load_json(ROOT / r["page_evidence"])) for r in added)
    check("GALLERY-IDENTITY", "كل صورة جديدة مرتبطة برابط معرض فعلي في HTML وهوية الإعلان نفسها", identity_ok)
    mutable = {"n_images", "n_overview_images", "usable_recognition", "usable_make_recognition", "usable_color", "usable_body_type", "usable_visual_search",
               "recognition_status", "body_type_std", "body_label_status", "body_label_evidence"}
    check("LISTINGS", "الإضافة لم تغير سعرًا أو ممشى أو مصدرًا أو قسمًا أو معرفًا تاريخيًا",
          {r["listing_uid"] for r in old_listings} == set(parents)
          and all(all(parents[r["listing_uid"]][k] == v for k, v in r.items() if k not in mutable) for r in old_listings))
    @lru_cache(maxsize=256)
    def thumb(path):
        with Image.open(ROOT / path) as im:
            return np.asarray(ImageOps.exif_transpose(im).convert("RGB").resize((32, 32), Image.Resampling.LANCZOS), dtype=np.float32)
    unsafe = []
    for r in added:
        for other in images:
            if other["image_uid"] == r["image_uid"] or other["group_id"] == r["group_id"]: continue
            exact = r["sha256"] == other["sha256"]
            if not exact and ((int(r["phash"], 16) ^ int(other["phash"], 16)).bit_count() > 3 or
                              (int(r["dhash"], 16) ^ int(other["dhash"], 16)).bit_count() > 4): continue
            if exact or float(np.abs(thumb(r["relative_path"]) - thumb(other["relative_path"])).mean()) <= 6:
                unsafe.append([r["image_uid"], other["image_uid"]])
    check("DUPLICATES", "الصور الجديدة لا تربط مجموعات مستقلة سابقة بتكرار مطابق أو اشتباه بصري شديد", not unsafe, unsafe[:10])
    old_zero = sum(int(r["n_overview_images"]) == 0 for r in old_listings)
    new_zero = sum(int(r["n_overview_images"]) == 0 for r in listings)
    check("COVERAGE", "تحسن تغطية مناظر السيارة مطابق للصور المعتمدة والعدادات",
          all(int(next(x for x in old_listings if x["listing_uid"] == r["listing_uid"])["n_overview_images"]) == 0 for r in added)
          and old_zero - new_zero == len({r["listing_uid"] for r in added})
          and summary["supplemental_images"]["remaining_no_overview"] == new_zero,
          {"before": old_zero, "after": new_zero})
    return checks
