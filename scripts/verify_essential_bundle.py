"""تحقق داخلي من النسخة المختصرة؛ لا يضيف ملفات إلى مجلد المستخدم."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

def read(path):
    with path.open(encoding="utf-8-sig",newline="") as stream:
        return list(csv.DictReader(stream))

def verify(folder, processed_dir=None):
    from build_essential_bundle import ROOT, LISTING_FILE, IMAGE_FILE, LISTING_COLUMNS, IMAGE_COLUMNS, image_path
    folder=folder.resolve()
    if {p.name for p in folder.iterdir()}!={LISTING_FILE,IMAGE_FILE,"الصور"}:
        raise ValueError("المجلد يجب أن يحتوي جدولين ومجلد الصور فقط")
    listings=read(folder/LISTING_FILE); images=read(folder/IMAGE_FILE)
    processed_dir=Path(processed_dir).resolve() if processed_dir else ROOT/"data/processed"
    original_listings=read(processed_dir/"listings.csv")
    original_images=read(processed_dir/"images.csv")
    expected_listings=[{key:row[key] for key in LISTING_COLUMNS} for row in original_listings]
    expected_images=[{key:(image_path(row) if key=="relative_path" else row[key]) for key in IMAGE_COLUMNS} for row in original_images]
    if listings!=expected_listings or images!=expected_images:
        raise ValueError("قيم الجداول تختلف عن البيانات النهائية؛ يسمح فقط باختصار الأعمدة وتعديل مسار الصورة")
    accepted={row["listing_uid"]:row for row in listings}
    paths={row["relative_path"] for row in images}
    if len(accepted)!=len(listings) or len(paths)!=len(images): raise ValueError("تكرار معرف أو مسار")
    actual={p.relative_to(folder).as_posix() for p in (folder/"الصور").rglob("*") if p.is_file()}
    if actual!=paths: raise ValueError("صور زائدة أو ناقصة")
    def check_image(row):
        relative=Path(row["relative_path"])
        if relative.is_absolute() or ".." in relative.parts: raise ValueError("مسار غير صالح")
        path=folder/relative
        h=hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda:stream.read(1024*1024),b""): h.update(block)
        if h.hexdigest()!=row["sha256"]: raise ValueError("بصمة صورة مختلفة")
        if row["listing_uid"] not in accepted or row["split"]!=accepted[row["listing_uid"]]["split"]:
            raise ValueError("ربط إعلان أو تقسيم غير صحيح")
    with ThreadPoolExecutor(max_workers=4) as executor:
        for _ in executor.map(check_image,images): pass
    counts=Counter(row["listing_uid"] for row in images)
    if any(int(row["n_images"])!=counts[row["listing_uid"]] for row in listings): raise ValueError("عدد صور الإعلان غير مطابق")
    seen={}
    for row in images:
        if seen.setdefault(row["sha256"],row["split"])!=row["split"]: raise ValueError("تسرب الصور بين الأقسام")
    return {"accepted_listings":len(listings),"accepted_images":len(images),"top_level_entries":3,"passed":True}

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,default=Path(__file__).resolve().parents[1]/"الملفات_المهمة")
    parser.add_argument("--processed-dir",type=Path,default=None)
    args=parser.parse_args()
    print(json.dumps(verify(args.root,args.processed_dir),ensure_ascii=True))

if __name__=="__main__":
    main()
