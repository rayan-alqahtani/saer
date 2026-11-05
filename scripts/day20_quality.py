"""مجموعات التكرار الحقيقي والمشتبه به؛ لا تمثل أحكامًا بالاحتيال."""
from collections import defaultdict
from functools import lru_cache
import numpy as np
from PIL import Image, ImageOps
from day20_common import ROOT, text

class UnionFind:
    def __init__(self, keys=()):
        self.parent = {k: k for k in keys}

    def find(self, key):
        self.parent.setdefault(key, key)
        if self.parent[key] != key:
            self.parent[key] = self.find(self.parent[key])
        return self.parent[key]

    def union(self, left, right):
        a, b = self.find(left), self.find(right)
        if a != b:
            self.parent[max(a, b)] = min(a, b)

@lru_cache(maxsize=2048)
def thumbnail(path):
    with Image.open(ROOT / path) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        return np.asarray(image.resize((32, 32), Image.Resampling.LANCZOS), dtype=np.float32)

def duplicate_groups(images, listings):
    """تجميع احترازي قبل التقسيم. الاشتباه البصري لا يعد تطابقًا مثبتًا."""
    image_groups = UnionFind(r["image_uid"] for r in images)
    listing_groups = UnionFind(r["listing_uid"] for r in listings)
    by_listing = {r["listing_uid"]: r for r in listings}
    by_hash = defaultdict(list)
    for image in images:
        if image["is_valid"]:
            by_hash[image["sha256"]].append(image)
    pairs = []
    conflicts = set()
    for digest, group in by_hash.items():
        labels = {(by_listing[r["listing_uid"]]["make"], by_listing[r["listing_uid"]]["model"]) for r in group if r["listing_uid"] in by_listing}
        if len(labels) > 1:
            conflicts.update(r["image_uid"] for r in group)
        for other in group[1:]:
            first = group[0]
            image_groups.union(first["image_uid"], other["image_uid"])
            listing_groups.union(first["listing_uid"], other["listing_uid"])
            pairs.append({"left_image_uid": first["image_uid"], "right_image_uid": other["image_uid"], "left_listing_uid": first["listing_uid"], "right_listing_uid": other["listing_uid"], "kind": "exact_sha256", "pixel_mae": 0, "same_make": len({label[0] for label in labels}) == 1})
    representatives = [sorted(g, key=lambda r: r["image_uid"])[0] for g in by_hash.values()]
    # أربعة أحزمة متساوية: لأي فرق <=3 بتات يوجد حزام مطابق كامل.
    buckets = defaultdict(list)
    seen_pairs = set()
    for index, image in enumerate(representatives):
        if not image.get("phash") or not image.get("dhash"):
            continue
        p = int(image["phash"], 16)
        candidate_ids = set()
        for band in range(4):
            candidate_ids.update(buckets[(band, (p >> (band * 16)) & 0xffff)])
        for idx in candidate_ids:
            other = representatives[idx]
            if other["listing_uid"] == image["listing_uid"]:
                continue
            if (int(other["phash"], 16) ^ p).bit_count() > 3:
                continue
            if (int(other["dhash"], 16) ^ int(image["dhash"], 16)).bit_count() > 4:
                continue
            key = tuple(sorted((image["image_uid"], other["image_uid"])))
            if key in seen_pairs:
                continue
            seen_pairs.add(key)
            mae = float(np.abs(thumbnail(image["relative_path"]) - thumbnail(other["relative_path"])).mean())
            if mae > 6:
                continue
            a, b = by_listing.get(image["listing_uid"], {}), by_listing.get(other["listing_uid"], {})
            same_make = a.get("make") == b.get("make")
            # أي زوج شديد التشابه يبقى في نفس القسم حتى لو كانت هويته غير محسومة.
            image_groups.union(image["image_uid"], other["image_uid"])
            listing_groups.union(image["listing_uid"], other["listing_uid"])
            pairs.append({"left_image_uid": image["image_uid"], "right_image_uid": other["image_uid"], "left_listing_uid": image["listing_uid"], "right_listing_uid": other["listing_uid"], "kind": "near_duplicate_suspected", "pixel_mae": round(mae, 4), "same_make": same_make})
            if not same_make:
                conflicts.update((image["image_uid"], other["image_uid"]))
        for band in range(4):
            buckets[(band, (p >> (band * 16)) & 0xffff)].append(index)
    # تطابق المواصفات وحده اشتباه في الإعلان؛ يجمع احترازيًا ولا يحذف السجلات.
    structured = defaultdict(list)
    for row in listings:
        fields = [row.get(k) for k in ["make", "model", "year", "sar_price", "km_mileage"]]
        if row.get("condition") == "used" and all(v is not None and text(v) for v in fields):
            structured[tuple(fields)].append(row["listing_uid"])
    structured_pairs = []
    for group in structured.values():
        for other in group[1:]:
            listing_groups.union(group[0], other)
            structured_pairs.append({"left_listing_uid": group[0], "right_listing_uid": other, "kind": "same_verified_structured_fields"})
    return image_groups, listing_groups, pairs, structured_pairs, conflicts
