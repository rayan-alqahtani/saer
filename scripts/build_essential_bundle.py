"""تجهيز زبدة مخرجات اليوم: جدولان والصور المعتمدة فقط، دون تقارير."""
from __future__ import annotations
import csv
import argparse
import hashlib
import json
import shutil
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LISTING_FILE = "الداتا_المحدثة.csv"
IMAGE_FILE = "ربط_الصور.csv"
LISTING_COLUMNS = ["listing_uid", "source_dataset", "source_listing_id", "external_listing_id", "make", "model", "model_std", "year", "trim", "condition", "sar_price", "km_mileage", "city", "transmission", "type_fuel", "color", "body_type", "listing_date", "url_source", "date_collection", "n_images", "group_id", "split", "usable_price", "usable_recognition", "rec_label"]
IMAGE_COLUMNS = ["image_uid", "listing_uid", "relative_path", "width", "height", "sha256", "label_make", "label_model", "label_color", "label_body_type", "image_kind", "is_exterior", "exterior_status", "usable_recognition", "usable_visual_search", "split", "group_id", "duplicate_group", "rec_label", "source_url"]
LISTING_COLUMNS += ["data_quality_flags", "recognition_status"]
LISTING_COLUMNS += ["mileage_source_status", "current_source_mileage", "source_verified_at", "trim_status"]
LISTING_COLUMNS += ["usable_make_recognition"]
IMAGE_COLUMNS += ["recognition_status"]
IMAGE_COLUMNS += ["usable_make_recognition", "make_recognition_status"]
LISTING_COLUMNS += ["trim_std", "color_std", "body_type_std", "color_label_status", "body_label_status", "price_status", "n_overview_images", "usable_visual_search", "usable_color", "usable_body_type"]
IMAGE_COLUMNS += ["content_kind", "content_review_status", "label_color_std", "label_body_type_std", "color_label_status", "body_label_status", "usable_color", "usable_body_type", "color_status", "body_type_status"]
LISTING_COLUMNS += ["recognition_route", "flag_source_unavailable", "flag_high_annual_mileage", "flag_price_iqr_outlier", "flag_mileage_exact_100", "flag_young_low_mileage"]
IMAGE_COLUMNS += ["recognition_route"]

def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))

def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def image_path(row):
    return (Path("الصور") / row["source_dataset"] / row["source_listing_id"] / Path(row["relative_path"]).name).as_posix()

def verify(folder, processed_dir=None):
    from verify_essential_bundle import verify as verify_current
    return verify_current(folder, processed_dir)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh", action="store_true", help="تحديث الجدولين مع حفظ السابق وإبقاء الصور المطابقة")
    parser.add_argument("--processed-dir", type=Path, default=ROOT / "data/processed", help="مصدر الجداول المعتمدة؛ يدعم تجهيز النسخة المرحلة قبل النشر")
    args = parser.parse_args()
    processed_dir = args.processed_dir.resolve()
    if args.refresh and processed_dir != ROOT / "data/processed":
        raise ValueError("لا تحديث لمجلد المستخدم من نسخة لم تنشر؛ جهز حزمة مرحلية مستقلة أولًا")
    stage = ROOT / "الملفات_المهمة" if args.refresh else ROOT / "data/staging" / ("essential_compact_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f"))
    if not args.refresh:
        stage.mkdir(parents=True, exist_ok=False)
    elif {p.name for p in stage.iterdir()} != {LISTING_FILE, IMAGE_FILE, "الصور"}:
        raise ValueError("مجلد التحديث ليس النسخة المختصرة المعروفة")
    listings = read_csv(processed_dir / "listings.csv")
    images = read_csv(processed_dir / "images.csv")
    assert len({image_path(r) for r in images}) == len(images), "تعارض أسماء الصور"
    if args.refresh:
        old_images = read_csv(stage / IMAGE_FILE)
        if {(r["relative_path"], r["sha256"]) for r in old_images} != {(image_path(r), r["sha256"]) for r in images}:
            raise ValueError("مجموعة الصور تغيرت؛ يلزم بناء نسخة مرحلية مستقلة")
    print("Preparing " + str(len(listings)) + " listings and " + str(len(images)) + " images", flush=True)
    def copy_image(row):
        source = ROOT / row["relative_path"]
        destination = stage / image_path(row)
        if not args.refresh:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
        if digest(destination) != row["sha256"]:
            raise ValueError("صورة لم تنسخ كاملة: " + row["image_uid"])
        return True
    with ThreadPoolExecutor(max_workers=4) as executor:
        for index, _ in enumerate(executor.map(copy_image, images), 1):
            if index % 3000 == 0:
                print("Copied " + str(index) + "/" + str(len(images)), flush=True)
    for name, rows, fields in [(LISTING_FILE,listings,LISTING_COLUMNS),(IMAGE_FILE,images,IMAGE_COLUMNS)]:
        import io
        from day20_common import save_bytes
        stream = io.StringIO(newline="")
        writer=csv.DictWriter(stream,fieldnames=fields,lineterminator="\n")
        writer.writeheader()
        for row in rows:
            result={key:row[key] for key in fields}
            if name==IMAGE_FILE: result["relative_path"]=image_path(row)
            writer.writerow(result)
        save_bytes(stage / name, stream.getvalue().encode("utf-8-sig"))
    result=verify(stage, processed_dir)
    state={"stage":str(stage),"processed_data_path":str(processed_dir),"published":args.refresh,"counts":result,"verified_at":datetime.now(timezone.utc).isoformat(timespec="seconds")}
    from day20_common import save_json
    save_json(ROOT/"data/audit/day20/compact_bundle.json",state)
    print(json.dumps(state,ensure_ascii=True),flush=True)

if __name__=="__main__":
    main()
