"""استرجاع النواقص من صفحات عامة مطابقة للمعرّف، مع دليل وشبكة محدودة السرعة.

لا يستخدم API ولا تسجيل دخول. يتوقف المضيف عند 403/429، ولا يغير الخام.
"""
from __future__ import annotations
from collections import defaultdict
import io
import gzip
import hashlib
import json
from pathlib import Path
import queue
import re
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from PIL import Image
from day20_common import *

UA = "SaerAcademic/1.0"
FORMATS = {"WEBP": ".webp", "JPEG": ".jpg", "PNG": ".png"}

class PublicClient:
    def __init__(self, compressed=False):
        self.compressed = compressed
        self.last = defaultdict(float)
        self.stopped = {}
        self.rules = {}
        self.locks = defaultdict(threading.Lock)
        self.log_lock = threading.Lock()

    def log(self, record):
        record["at"] = now()
        with self.log_lock:
            path = RECOVERY / "network.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def get(self, url, robots=False):
        parsed = urllib.parse.urlsplit(url)
        host = parsed.hostname
        if parsed.scheme != "https" or host not in {"example.invalid", "cdn.example.invalid"}:
            return None, {"status": "rejected_host", "url": url}
        if host in self.stopped:
            return None, {"status": "host_stopped", "reason": self.stopped[host], "url": url}
        if not robots and not self.allowed(url):
            return None, {"status": "robots_disallowed", "url": url}
        with self.locks[host]:
            time.sleep(max(0, 2.1 - (time.monotonic() - self.last[host])))
            for attempt in range(2):
                self.last[host] = time.monotonic()
                try:
                    headers = {"User-Agent": UA, "Accept": "text/html,image/webp,image/*;q=0.8,*/*;q=0.5"}
                    if self.compressed:
                        headers["Accept-Encoding"] = "gzip"
                    request = urllib.request.Request(url, headers=headers)
                    with urllib.request.urlopen(request, timeout=25) as response:
                        final = urllib.parse.urlsplit(response.url)
                        if final.hostname != host:
                            raise ValueError("unexpected_redirect_host")
                        body = response.read(6_000_001)
                        if len(body) > 6_000_000:
                            raise ValueError("response_too_large")
                        wire_bytes = len(body)
                        if response.headers.get("Content-Encoding", "").lower() == "gzip":
                            with gzip.GzipFile(fileobj=io.BytesIO(body)) as gz:
                                body = gz.read(6_000_001)
                            if len(body) > 6_000_000:
                                raise ValueError("decompressed_response_too_large")
                        result = {"wire_bytes": wire_bytes, "status": response.status, "url": url, "final_url": response.url, "bytes": len(body), "content_type": response.headers.get("Content-Type", ""), "sha256": hashlib.sha256(body).hexdigest()}
                        self.log(result)
                        return body, result
                except urllib.error.HTTPError as error:
                    body = error.read(30000)
                    if error.headers.get("Content-Encoding", "").lower() == "gzip":
                        try:
                            with gzip.GzipFile(fileobj=io.BytesIO(body)) as gz:
                                body = gz.read(30001)
                        except (OSError, EOFError):
                            pass  # يبقى رفض المضيف نافذًا إن تعذر فك رسالة الخطأ.
                    # هذا رد S3 لملف robots غير الموجود، كما في السكربت الأصلي.
                    no_robot_file = robots and host == "cdn.example.invalid" and error.code == 403 and b"<Code>AccessDenied</Code>" in body
                    result = {"status": error.code, "url": url, "missing_cdn_robots": no_robot_file}
                    self.log(result)
                    if error.code in (403, 429) and not no_robot_file:
                        self.stopped[host] = error.code
                    if no_robot_file:
                        return b"", result
                    if error.code in (500, 502, 503, 504) and attempt == 0:
                        time.sleep(2.1)
                        continue
                    return None, result
                except Exception as error:
                    result = {"status": "network_error", "url": url, "error": str(error), "attempt": attempt + 1}
                    self.log(result)
                    if attempt == 0:
                        time.sleep(2.1)
            return None, result

    def allowed(self, url):
        parsed = urllib.parse.urlsplit(url)
        rules = self.rules.get(parsed.hostname)
        if rules is None:
            return False
        target = parsed.path + ("?" + parsed.query if parsed.query else "")
        matches = []
        for allow, pattern in rules:
            end = pattern.endswith("$")
            p = pattern[:-1] if end else pattern
            regex = "^" + re.escape(p).replace(r"\*", ".*") + ("$" if end else "")
            if re.search(regex, target):
                matches.append((len(p.replace("*", "")), allow))
        return max(matches, default=(0, True))[1]

    def init_robots(self):
        for host in ["example.invalid", "cdn.example.invalid"]:
            body, status = self.get(f"https://{host}/robots.txt", robots=True)
            if body is None:
                self.rules[host] = None
                continue
            save_bytes(RECOVERY / "robots" / (host + ".txt"), body)
            rules = []
            applies = False
            for line in body.decode("utf-8", "replace").splitlines():
                line = line.split("#", 1)[0].strip()
                if ":" not in line:
                    continue
                key, value = [s.strip() for s in line.split(":", 1)]
                if key.lower() == "user-agent":
                    applies = value == "*" or value.lower() in UA.lower()
                elif applies and key.lower() in {"allow", "disallow"} and value:
                    rules.append((key.lower() == "allow", value))
            self.rules[host] = rules
            save_json(RECOVERY / "robots" / (host + ".json"), {"fetched_at": now(), "response": status, "rules": rules})

def basename(url):
    return urllib.parse.urlsplit(url).path.rsplit("/", 1)[-1]

def prepare_queue():
    if not (AUDIT / "raw_manifest.json").exists():
        raise SystemExit("لم تكتمل النسخة الخام المتحققة.")
    jobs = []
    for source, filename in SOURCES.items():
        rows = read_csv(RAW / source / filename)
        downloaded = read_csv(RAW / source / "car_images_downloaded.csv")
        catalog = read_csv(RAW / source / "car_images.csv")
        downloaded_ids = {r["id_listing"] for r in downloaded}
        keys = {(r["id_listing"], int(r["image_no"])) for r in downloaded}
        missing_ids = {r["id_listing"] for r in catalog if (r["id_listing"], int(r["image_no"])) not in keys}
        cached = cache_rows(source)
        for row in rows:
            reasons = recovery_reasons(effective_initial(row, cached.get(row["external_listing_id"], {})), downloaded_ids, missing_ids)
            if reasons:
                jobs.append({"source_dataset": source, "id_listing": row["id_listing"], "external_listing_id": row["external_listing_id"], "url": row["url_source"], "reasons": reasons, "priority": 0 if "no_downloaded_image" in reasons else 1 if "missing_sar_price" in reasons else 2})
    jobs.sort(key=lambda r: (r["priority"], r["source_dataset"], r["id_listing"]))
    save_json(AUDIT / "recovery_queue.json", jobs)
    print(json.dumps({"recovery_pages": len(jobs)}, ensure_ascii=False), flush=True)

def run():
    jobs = load_json(AUDIT / "recovery_queue.json")
    if jobs is None:
        raise SystemExit("شغل الجرد أولًا؛ لا استرجاع دون قائمة مشاكل.")
    state_path = RECOVERY / "manifest.json"
    state = load_json(state_path, {"created_at": now(), "pages": {}, "images": [], "errors": []})
    if state.get("complete"):
        all_saved = all((ROOT / row["relative_path"]).is_file() for row in state["images"]) and all((ROOT / page["parsed_path"]).is_file() for page in state["pages"].values() if page.get("status") == "ok")
        if all_saved:
            print(json.dumps({"complete": True, "resumed_without_network": True, "pages": len(state["pages"]), "recovered_images": len(state["images"])}, ensure_ascii=False), flush=True)
            return
        raise SystemExit("بعض أدلة الاسترجاع المكتمل مفقودة؛ لا تستبدل السجل، تحقق من الملفات أولًا.")
    client = PublicClient()
    client.init_robots()
    state_lock = threading.Lock()
    image_jobs = queue.Queue()
    downloaded_by_source = {source: read_csv(RAW / source / "car_images_downloaded.csv") for source in SOURCES}
    catalog_by_source = {source: read_csv(RAW / source / "car_images.csv") for source in SOURCES}
    done_image_urls = {row["url"] for row in state["images"] if (ROOT / row["relative_path"]).is_file()}
    queued_urls = set(done_image_urls)

    def persist():
        state["updated_at"] = now()
        state["host_stops"] = client.stopped
        save_json(state_path, state)

    def image_worker():
        processed = 0
        while True:
            task = image_jobs.get()
            try:
                if task is None:
                    return
                body, result = client.get(task["url"])
                if body is None:
                    with state_lock:
                        state["errors"].append({**task, "response": result, "entity_type": "image"})
                        persist()
                    continue
                with Image.open(io.BytesIO(body)) as image:
                    image.load()
                    fmt = image.format
                    dimensions = image.size
                if fmt not in FORMATS or min(dimensions) < 150:
                    raise ValueError("unsupported_or_low_resolution_image")
                token = hashlib.sha256(task["url"].encode()).hexdigest()[:16]
                path = RECOVERY / "images" / task["source_dataset"] / task["id_listing"] / ("r_" + token + FORMATS[fmt])
                save_bytes(path, body)
                with state_lock:
                    state["images"].append({**task, "relative_path": relative(path), "sha256": hashlib.sha256(body).hexdigest(), "width": dimensions[0], "height": dimensions[1], "fetched_at": now()})
                    persist()
                processed += 1
                if processed % 20 == 0:
                    print(f"صور مسترجعة: {len(state['images'])}", flush=True)
            except Exception as error:
                with state_lock:
                    state["errors"].append({**(task or {}), "entity_type": "image", "error": str(error), "at": now()})
                    persist()
            finally:
                image_jobs.task_done()

    worker = threading.Thread(target=image_worker, daemon=True)
    worker.start()
    for index, job in enumerate(jobs, 1):
        source, listing_id, sid = job["source_dataset"], job["id_listing"], job["external_listing_id"]
        key = uid(source, listing_id)
        parsed_path = RECOVERY / "pages" / source / (sid + ".json")
        html_path = parsed_path.with_suffix(".html")
        existing = state["pages"].get(key, {})
        parsed = None
        if existing.get("status") == "ok" and parsed_path.exists():
            parsed = load_json(parsed_path)
        elif existing and existing.get("status") in {403, 404, 410, 429, "robots_disallowed", "identity_mismatch", "source_redirected_or_unavailable", "'postDetails'", "source_deleted", "cash_price_mismatch"}:
            continue
        else:
            sample = RECOVERY / "samples" / (sid + ".html")
            if sample.exists():
                body, response = sample.read_bytes(), {"status": 200, "url": job["url"], "from_sample": True}
            else:
                body, response = client.get(job["url"])
            if body is None:
                with state_lock:
                    state["pages"][key] = {"status": response["status"], "response": response, "at": now(), "reasons": job["reasons"]}
                    persist()
                continue
            save_bytes(html_path, body)
            try:
                parsed = public_page_data(body.decode("utf-8", "replace"), sid)
                parsed.update(source_dataset=source, id_listing=listing_id, external_listing_id=sid, fetched_at=now(), html_path=relative(html_path), source_url=job["url"], html_sha256=hashlib.sha256(body).hexdigest())
                save_json(parsed_path, parsed)
                with state_lock:
                    state["pages"][key] = {"status": "ok", "parsed_path": relative(parsed_path), "at": now(), "reasons": job["reasons"]}
                    persist()
            except Exception as error:
                with state_lock:
                    state["pages"][key] = {"status": str(error), "html_path": relative(html_path), "at": now(), "reasons": job["reasons"]}
                    persist()
                continue
        if parsed and any(r in job["reasons"] for r in ["no_downloaded_image", "failed_image_slot"]):
            existing_rows = [r for r in downloaded_by_source[source] if r["id_listing"] == listing_id]
            existing_names = {basename(r["url_image"]) for r in existing_rows}
            original_keys = {(r["id_listing"], int(r["image_no"])) for r in existing_rows}
            missing = [r for r in catalog_by_source[source] if r["id_listing"] == listing_id and (listing_id, int(r["image_no"])) not in original_keys]
            missing_names = {basename(r["url_image"]): int(r["image_no"]) for r in missing}
            needed = max(5 if not existing_rows else len(missing), 1)
            candidates = [r for r in parsed["photos"] if basename(r["url"]) not in existing_names]
            candidates.sort(key=lambda r: (basename(r["url"]) not in missing_names, r["image_kind"] != "exterior"))
            for photo in candidates[:needed]:
                if photo["url"] not in queued_urls:
                    queued_urls.add(photo["url"])
                    image_jobs.put({**photo, "source_dataset": source, "id_listing": listing_id, "listing_uid": key, "external_listing_id": sid, "page_evidence": relative(parsed_path), "original_slot_no": missing_names.get(basename(photo["url"]), "")})
        if index % 20 == 0:
            print(json.dumps({"pages_processed": index, "pages_total": len(jobs), "pages_saved": sum(v["status"] == "ok" for v in state["pages"].values()), "images_saved": len(state["images"]), "image_queue": image_jobs.qsize(), "host_stops": client.stopped}, ensure_ascii=False), flush=True)
    image_jobs.put(None)
    image_jobs.join()
    worker.join()
    with state_lock:
        state["complete"] = True
        persist()
    print(json.dumps({"complete": True, "pages": len(state["pages"]), "pages_ok": sum(v["status"] == "ok" for v in state["pages"].values()), "recovered_images": len(state["images"]), "errors": len(state["errors"]), "host_stops": client.stopped}, ensure_ascii=False), flush=True)

if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    if "--prepare-queue" in sys.argv:
        prepare_queue()
    else:
        run()
