"""جرد قبل أي تنظيف أو دمج، مع نسخة خام متحققة ومخطط أخطاء مستقل لكل مصدر."""
from __future__ import annotations
import ast
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import shutil
import sys
import zipfile
import numpy as np
from PIL import Image, ImageOps
from openpyxl import load_workbook
from day20_common import *

def snapshot():
    manifest_path = AUDIT / "raw_manifest.json"
    if manifest_path.exists():
        print("النسخة الخام موجودة؛ لن يعاد نسخها أو استبدالها.", flush=True)
        return load_json(manifest_path)
    jobs = [(source, path) for source in SOURCES for path in sorted((ROOT / source).rglob("*")) if path.is_file()]
    def one(job):
        source, original = job
        target = RAW / source / original.relative_to(ROOT / source)
        target.parent.mkdir(parents=True, exist_ok=True)
        before = sha256(original)
        if target.exists():
            if sha256(target) != before:
                raise ValueError("النسخة الخام مختلفة، لن تستبدل: " + relative(target))
        else:
            shutil.copy2(original, target)
        if sha256(target) != before:
            raise ValueError("snapshot_hash_mismatch: " + str(target))
        return {"original_path": relative(original), "raw_path": relative(target), "bytes": original.stat().st_size, "sha256": before}
    rows = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        for result in pool.map(one, jobs):
            rows.append(result)
            if len(rows) % 2000 == 0:
                print(f"نسخ وتحقق: {len(rows)}/{len(jobs)}", flush=True)
    result = {"created_at": now(), "files": rows, "total_files": len(rows), "total_bytes": sum(r["bytes"] for r in rows)}
    save_json(manifest_path, result)
    return result

def image_stats(path):
    result = {"relative_path": relative(path), "sha256": sha256(path), "is_valid": False, "error": "", "width": "", "height": "", "dhash": "", "phash": "", "low_resolution": False}
    try:
        with Image.open(path) as im:
            im.load()
            result.update(width=im.width, height=im.height, image_format=im.format)
            im = ImageOps.exif_transpose(im).convert("RGB")
            gray = np.asarray(im.convert("L").resize((9, 8), Image.Resampling.LANCZOS), dtype=np.int16)
            bits = gray[:, 1:] > gray[:, :-1]
            result["dhash"] = f"{int.from_bytes(np.packbits(bits).tobytes(), 'big'):016x}"
            pixels = np.asarray(im.convert("L").resize((32, 32), Image.Resampling.LANCZOS), dtype=np.float64)
            n = np.arange(32)
            mat = np.cos(np.pi / 32 * (n[None, :] + .5) * n[:, None])
            dct = (mat @ pixels @ mat.T)[:8, :8]
            threshold = np.median(dct.reshape(-1)[1:])
            result["phash"] = f"{int.from_bytes(np.packbits(dct > threshold).tobytes(), 'big'):016x}"
            result["low_resolution"] = min(im.size) < 150
            result["pixel_std"] = round(float(pixels.std()), 3)
            result["is_valid"] = True
    except Exception as error:
        result["error"] = str(error)
    return result

def inventory():
    if (AUDIT / "before.json").exists() and (AUDIT / "raw_manifest.json").exists() and (AUDIT / "recovery_queue.json").exists():
        result = load_json(AUDIT / "before.json")
        print(json.dumps({"inventory_already_complete": True, "raw_listings": result["raw_listings"], "raw_images": result["raw_images"], "raw_integrity_checked_separately_by_validate": True}, ensure_ascii=False), flush=True)
        return
    snapshot()
    all_images = []
    source_summaries = {}
    recovery_queue = []
    listing_issues = []
    for source, main_name in SOURCES.items():
        base = RAW / source
        listings = read_csv(base / main_name)
        catalog = read_csv(base / "car_images.csv")
        downloaded = read_csv(base / "car_images_downloaded.csv")
        cached = cache_rows(source)
        downloaded_ids = {r["id_listing"] for r in downloaded}
        downloaded_keys = {(r["id_listing"], int(r["image_no"])) for r in downloaded}
        missing_slots = [r for r in catalog if (r["id_listing"], int(r["image_no"])) not in downloaded_keys]
        missing_ids = {r["id_listing"] for r in missing_slots}
        lid = {r["id_listing"] for r in listings}
        xlsx = next(base.glob("external_member*.xlsx"))
        wb = load_workbook(xlsx, read_only=True, data_only=True)
        sheet_details = [{"name": s.title, "rows_including_header": s.max_row, "columns": s.max_column, "headers": list(next(s.iter_rows(values_only=True), []))} for s in wb]
        sheet = wb["cars_clean"]
        it = sheet.iter_rows(values_only=True)
        headers = next(it)
        excel_rows = [dict(zip(headers, row)) for row in it]
        excel_ids = {str(r.get("id_listing")) for r in excel_rows}
        excel_differences = []
        by_id = {str(r["id_listing"]): r for r in excel_rows}
        for row in listings:
            ex = by_id.get(row["id_listing"], {})
            for key in row:
                a, b = text(row[key]), text(ex.get(key))
                if a != b and not (number(a) is not None and number(a) == number(b)):
                    excel_differences.append({"id_listing": row["id_listing"], "field": key, "csv": a, "xlsx": b})
        wb.close()
        script_checks = []
        for path in sorted(base.rglob("*.py")):
            error = ""
            try:
                ast.parse(path.read_text(encoding="utf-8-sig"))
            except Exception as exc:
                error = str(exc)
            script_checks.append({"path": relative(path), "sha256": sha256(path), "syntax_valid": not error, "error": error})
        archive_checks = []
        for path in sorted(base.rglob("*.zip")):
            with zipfile.ZipFile(path) as archive:
                archive_checks.append({"path": relative(path), "members": len(archive.infolist()), "crc_error": archive.testzip()})
        for row in listings:
            initial = effective_initial(row, cached.get(row["external_listing_id"], {}))
            reasons = recovery_reasons(initial, downloaded_ids, missing_ids)
            if reasons:
                recovery_queue.append({"source_dataset": source, "id_listing": row["id_listing"], "external_listing_id": row["external_listing_id"], "url": row["url_source"], "reasons": reasons, "priority": 0 if "no_downloaded_image" in reasons else 1 if "missing_sar_price" in reasons else 2})
            for key, value in row.items():
                if not text(value):
                    listing_issues.append({"source_dataset": source, "listing_uid": uid(source, row["id_listing"]), "field": key, "issue_code": "missing_value", "original_value": ""})
        print(f"بدء فحص صور {source}", flush=True)
        paths = sorted((base / "images").rglob("*"))
        paths = [p for p in paths if p.is_file()]
        cached_inventory = {r["relative_path"]: r for r in read_csv(AUDIT / source / "images_before.csv")} if (AUDIT / source / "images_before.csv").exists() else {}
        jobs = [p for p in paths if relative(p) not in cached_inventory]
        with ThreadPoolExecutor(max_workers=4) as pool:
            stats = list(pool.map(image_stats, jobs))
        stats.extend(cached_inventory.values())
        for r in stats:
            path = Path(r["relative_path"])
            r["source_dataset"] = source
            r["source_listing_id"] = path.parent.name
            r["listing_uid"] = uid(source, path.parent.name)
            r["image_uid"] = r["listing_uid"] + ":" + path.stem
        save_csv(AUDIT / source / "images_before.csv", stats)
        save_csv(AUDIT / source / "missing_image_slots_before.csv", missing_slots, list(catalog[0]))
        save_csv(AUDIT / source / "excel_differences.csv", excel_differences, ["id_listing", "field", "csv", "xlsx"])
        all_images.extend(stats)
        summary = {
            "listings": len(listings), "columns": list(listings[0]), "missing": {k: sum(not text(r.get(k)) for r in listings) for k in listings[0]},
            "unique_listing_ids": len(lid), "unique_source_ids": len({r["external_listing_id"] for r in listings}),
            "makes": dict(Counter(r["make"] for r in listings)), "conditions": dict(Counter(r["condition"] for r in listings)),
            "images_on_disk": len(stats), "invalid_images": sum(str(r["is_valid"]).lower() != "true" for r in stats),
            "catalog_images": len(catalog), "missing_image_slots": len(missing_slots), "listings_without_images": len(lid - downloaded_ids),
            "image_orphans": sum(r["source_listing_id"] not in lid for r in stats),
            "workbook": relative(xlsx), "workbook_sheets": sheet_details, "excel_csv_id_match": excel_ids == lid,
            "excel_value_differences": len(excel_differences), "script_checks": script_checks, "archive_checks": archive_checks,
        }
        source_summaries[source] = summary
        save_json(AUDIT / source / "before.json", summary)
        print(json.dumps({"source": source, "listings": len(listings), "images": len(stats), "repair_pages": sum(q["source_dataset"] == source for q in recovery_queue)}, ensure_ascii=False), flush=True)
    recovery_queue.sort(key=lambda r: (r["priority"], r["source_dataset"], r["id_listing"]))
    save_json(AUDIT / "recovery_queue.json", recovery_queue)
    save_csv(AUDIT / "listing_issues_before.csv", listing_issues)
    save_csv(AUDIT / "images_before.csv", all_images)
    result = {"created_at": now(), "sources": source_summaries, "recovery_pages": len(recovery_queue), "raw_listings": sum(s["listings"] for s in source_summaries.values()), "raw_images": len(all_images)}
    save_json(AUDIT / "before.json", result)
    print(json.dumps({k: v for k, v in result.items() if k != "sources"}, ensure_ascii=False), flush=True)

if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    inventory()
