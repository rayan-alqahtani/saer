"""استكمال حقول المصدر والوسوم على دفعات قابلة للاستئناف، دون تغيير الخام."""
from __future__ import annotations
import argparse
import gzip
import hashlib
import json
import sys
from collections import Counter
from day20_common import *
from day20_recover import PublicClient

EVENTS = RECOVERY / "enrichment/events.jsonl"


def read_events():
    result = {}
    if EVENTS.exists():
        for line in EVENTS.read_text(encoding="utf-8").splitlines():
            if line.strip():
                item = json.loads(line)
                result[item["listing_uid"]] = item
    return result


def persist(record):
    EVENTS.parent.mkdir(parents=True, exist_ok=True)
    with EVENTS.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        stream.flush()


def confirmed_page(job, path):
    parsed = load_json(ROOT / path)
    if not parsed or str(parsed.get("external_listing_id")) != job["external_listing_id"] or not parsed.get("page_identity_verified"):
        return False
    return (ROOT / parsed["html_path"]).is_file()


def run(limit, budget_mb):
    queue = load_json(AUDIT / "enrichment_queue.json")
    if not queue:
        raise ValueError("قائمة الاستكمال غير موجودة")
    prior = read_events()
    original = load_json(RECOVERY / "manifest.json")["pages"]
    pilot = {r["listing_uid"]: r for r in load_json(AUDIT / "enrichment_pilot.json", {"results": []})["results"]}
    client = None
    downloads, wire_bytes = 0, 0
    for index, job in enumerate(queue, 1):
        key = job["listing_uid"]
        existing = prior.get(key)
        if existing and (existing["status"] != "ok" or confirmed_page(job, existing["parsed_path"])):
            continue
        record = {"listing_uid": key, "external_listing_id": job["external_listing_id"], "reasons": job["reasons"], "checked_at": now()}
        sample = pilot.get(key, {})
        saved = original.get(key, {})
        if sample.get("status") == "ok" and confirmed_page(job, sample["parsed_path"]):
            record.update(status="ok", parsed_path=sample["parsed_path"], from_verified_cache=True)
        elif saved.get("status") == "ok" and confirmed_page(job, saved["parsed_path"]):
            record.update(status="ok", parsed_path=saved["parsed_path"], from_verified_cache=True)
        elif sample and sample.get("status") not in {None, "ok"}:
            record.update(status=sample["status"], response=sample.get("response"), from_pilot=True)
        else:
            if downloads >= limit or wire_bytes >= budget_mb * 1024 ** 2:
                break
            if client is None:
                client = PublicClient(compressed=True)
                client.init_robots()
            body, response = client.get(job["url_source"])
            wire_bytes += response.get("wire_bytes", response.get("bytes", 0))
            downloads += 1
            record["response"] = response
            if body is None:
                record["status"] = response["status"]
                if response["status"] in {403, 429, "host_stopped", "robots_disallowed"}:
                    persist(record)
                    prior[key] = record
                    break
            else:
                html_path = RECOVERY / "enrichment/pages" / job["source_dataset"] / (job["external_listing_id"] + ".html.gz")
                save_bytes(html_path, gzip.compress(body, mtime=0))
                record["html_path"] = relative(html_path)
                try:
                    parsed = public_page_data(body.decode("utf-8", "replace"), job["external_listing_id"])
                    parsed.update(source_dataset=job["source_dataset"], id_listing=job["source_listing_id"],
                                  external_listing_id=job["external_listing_id"], fetched_at=now(), html_path=relative(html_path),
                                  source_url=job["url_source"], html_sha256=hashlib.sha256(body).hexdigest())
                    parsed_path = html_path.with_suffix("").with_suffix(".json")
                    save_json(parsed_path, parsed)
                    record.update(status="ok", parsed_path=relative(parsed_path))
                except Exception as error:
                    record["status"] = str(error)
        persist(record)
        prior[key] = record
        if downloads and downloads % 25 == 0:
            print(json.dumps({"checked": len(prior), "total": len(queue), "network_this_run": downloads,
                              "wire_MB_this_run": round(wire_bytes / 1024 ** 2, 2),
                              "status_counts": dict(Counter(r["status"] for r in prior.values()))}, ensure_ascii=False), flush=True)
    statuses = Counter(r["status"] for r in prior.values())
    result = {"updated_at": now(), "queue_size": len(queue), "checked": len(prior),
              "complete": len(prior) == len(queue), "status_counts": dict(statuses),
              "network_this_run": downloads, "wire_MB_this_run": round(wire_bytes / 1024 ** 2, 2),
              "host_stops": client.stopped if client else {}}
    save_json(RECOVERY / "enrichment/summary.json", result)
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return result


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--budget-mb", type=float, default=10)
    args = parser.parse_args()
    if args.limit < 0 or args.budget_mb <= 0:
        raise SystemExit("حدود الشبكة غير صالحة")
    run(args.limit, args.budget_mb)
