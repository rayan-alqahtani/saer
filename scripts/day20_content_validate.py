"""تحقق مستقل من البيانات المحفوظة؛ لا يستدعي دوال تطبيق السياسات."""
from collections import Counter, defaultdict
import json
from day20_common import ROOT, AUDIT, read_csv, load_json, sha256, relative


def content_checks(processed, listings, images, schema, summary):
    checks = []
    def check(code, description, okay, detail=""):
        checks.append(("T-C-" + code, description, bool(okay), detail))
    review = AUDIT / "content_review"
    source_listings = {r["listing_uid"]: r for r in read_csv(review / "baseline_listings.csv")}
    source_images = {r["image_uid"]: r for r in read_csv(review / "baseline_images.csv")}
    supplement = load_json(ROOT / summary["supplemental_images"]["manifest_path"]) if summary.get("supplemental_images") else None
    additional = {r["image_uid"]: r for r in supplement["accepted_images"]} if supplement else {}
    listing_changes = {"usable_price", "usable_recognition", "recognition_status", "usable_make_recognition"}
    image_changes = {"usable_recognition", "recognition_status", "usable_visual_search", "usable_make_recognition", "make_recognition_status"}
    if supplement: listing_changes.add("n_images")
    if summary.get("user_issues"): image_changes.add("exterior_status")
    check("PRESERVE", "كل قيم المصدر والتقسيم والصور الأصلية محفوظة؛ تغيرت الأهلية فقط",
          set(source_listings) == {r["listing_uid"] for r in listings}
          and set(source_images) | set(additional) == {r["image_uid"] for r in images}
          and all(all(r[k] == v for k, v in source_listings[r["listing_uid"]].items() if k not in listing_changes) for r in listings)
          and all(all(r[k] == v for k, v in source_images[r["image_uid"]].items() if k not in image_changes) for r in images if r["image_uid"] in source_images))
    decisions = {}
    decision_count = 0
    board_digests = {}
    decision_paths = sorted((review / "decisions").glob("*.csv"))
    if supplement: decision_paths.append(ROOT / supplement["accepted_decisions"])
    for p in decision_paths:
        for r in read_csv(p):
            decision_count += 1
            decisions[r["image_uid"]] = (r, relative(p))
            board = int(r["board"])
            if board not in board_digests:
                board_digests[board] = sha256(ROOT / r["board_path"] if r.get("board_path") else review / "boards" / f"board_{board:03}.jpg")
    check("REVIEW", "كل صورة لها قرار محتوى وحيد مرتبط ببصمتها ولوحتها",
          decision_count == len(images) and set(decisions) == set(source_images) | set(additional)
          and all(r.get("content_review_status") == "reviewed" and decisions[r["image_uid"]][0]["review_status"] == "reviewed"
                  and decisions[r["image_uid"]][0]["sha256"] == r["sha256"]
                  and decisions[r["image_uid"]][0]["content_kind"] == r["content_kind"]
                  and decisions[r["image_uid"]][1] == r["content_review_evidence"]
                  and decisions[r["image_uid"]][0]["board_sha256"] == board_digests[int(decisions[r["image_uid"]][0]["board"])] for r in images))
    kinds = defaultdict(set)
    for r in images: kinds[r["sha256"]].add(r["content_kind"])
    check("EXACT-CONTENT", "الصور ذات البصمة نفسها لها وسم محتوى واحد", all(len(v) == 1 for v in kinds.values()))
    bad = [r["image_uid"] for r in images if r["content_kind"] != "vehicle_overview" and any(
        r["usable_" + task] == "True" for task in ("recognition", "make_recognition", "visual_search", "color", "body_type"))]
    check("NO-DETAILS", "الوثائق والداخلية والتفاصيل مستبعدة من كل مهام منظر السيارة", not bad, bad[:10])
    # مقارنة مستقلة للخرائط المحفوظة؛ mapping صريح من وصف المصدر دون استنتاج من الموديل.
    color_map = schema["color_family_map"]
    body_map = schema["body_family_map"]
    body_decisions = {r["listing_uid"]: (r, relative(p)) for p in (review / "body_decisions").glob("*.csv") for r in read_csv(p)}
    new_body = {}
    if supplement:
        path = ROOT / supplement["body_decisions"]
        new_body = {r["listing_uid"]: (r, relative(path)) for r in read_csv(path)}
        body_decisions.update(new_body)
    issue_bodies = {}
    if summary.get("user_issues"):
        path = ROOT / summary["user_issues"]["body_overrides"]
        issue_bodies = {r["listing_uid"]: (r, relative(path)) for r in read_csv(path)}
        body_decisions.update(issue_bodies)
    body_evidence_ok, body_labels_ok = True, True
    expected_body_ids = {r["listing_uid"] for r in read_csv(review / "body_index.csv")}
    expected_body_ids.update(new_body)
    expected_body_ids.update(issue_bodies)
    for r in listings:
        decision, path = body_decisions.get(r["listing_uid"], (None, ""))
        body = body_map[r["body_type"]]
        if decision:
            body = "" if decision["body_visual"] == "unknown" else decision["body_visual"]
            actual = {(im["image_uid"], im["sha256"], im["relative_path"]) for im in images if im["listing_uid"] == r["listing_uid"]}
            evidence = {(im["image_uid"], im["sha256"], im["relative_path"]) for im in json.loads(decision["evidence_images"])}
            original_actual = {t for t in actual if t[0] not in additional}
            expected_evidence = actual if r["listing_uid"] in issue_bodies else original_actual
            body_evidence_ok &= (evidence <= actual if r["listing_uid"] in new_body else expected_evidence == evidence) and bool(evidence) and r["body_label_evidence"] == path
        body_labels_ok &= r["body_type_std"] == body
    check("BODY", "الوسم البصري مستقل عن المصدر ومثبت بصور الإعلان، وunknown لا يعوض بالتخمين",
          body_labels_ok and body_evidence_ok and set(body_decisions) == expected_body_ids,
          {"reviewed_listings": len(body_decisions)})
    check("COLOR-MAP", "عائلات اللون تطابق الخريطة الموثقة، واللون petrol الغامض يبقى بلا هدف",
          all(r["color_std"] == color_map[r["color"]] for r in listings) and color_map["petrol"] == "")
    by_id = {r["listing_uid"]: r for r in listings}
    check("LABEL-LINK", "وسوم الصور المشتقة تطابق إعلانها مع حفظ وسوم المصدر منفصلة",
          all(r["label_color_std"] == by_id[r["listing_uid"]]["color_std"] and
              r["label_body_type_std"] == by_id[r["listing_uid"]]["body_type_std"] for r in images))
    policy = schema["recognition_policy"]
    for task, field in (("color", "label_color_std"), ("body_type", "label_body_type_std")):
        sha_labels, groups = defaultdict(set), defaultdict(lambda: defaultdict(set))
        for r in images:
            if r["content_kind"] == "vehicle_overview" and r[field]: sha_labels[r["sha256"]].add(r[field])
        candidates, seen = [], set()
        for r in sorted(images, key=lambda x: x["image_uid"]):
            if r["content_kind"] != "vehicle_overview" or not r[field] or len(sha_labels[r["sha256"]]) != 1 or r["sha256"] in seen: continue
            seen.add(r["sha256"]); candidates.append(r)
            groups[r[field]][r["split"]].add(r["group_id"])
        supported = {label for label, folds in groups.items() if sum(len(v) for v in folds.values()) >= policy["min_total_groups"]
                     and all(len(folds[s]) >= policy["min_" + s + "_groups"] for s in ("train", "val", "test"))}
        expected = {r["image_uid"] for r in candidates if r[field] in supported}
        selected = {r["image_uid"] for r in images if r["usable_" + task] == "True"}
        parents = {r["listing_uid"] for r in images if r["image_uid"] in selected}
        reported = {r["label"] for r in read_csv(processed / "attribute_support.csv") if r["task"] == task and r["supported"] == "True"}
        check(task.upper(), "أهلية " + ("اللون" if task == "color" else "الهيكل") + " تعزل التعارض والنقص وقلة الدعم",
              expected == selected and reported == supported and all((r["usable_" + task] == "True") == (r["listing_uid"] in parents) for r in listings),
              {"labels": len(supported), "images": len(selected)})
    seen, search = set(), set()
    for r in sorted(images, key=lambda x: x["image_uid"]):
        if r["content_kind"] == "vehicle_overview" and r["sha256"] not in seen:
            seen.add(r["sha256"]); search.add(r["image_uid"])
    check("SEARCH", "فهرس البحث المهيأ مناظر سيارة فقط دون نسخ متطابقة ثنائيًا",
          {r["image_uid"] for r in images if r["usable_visual_search"] == "True"} == search)
    overview_counts = Counter(r["listing_uid"] for r in images if r["content_kind"] == "vehicle_overview")
    search_parents = {r["listing_uid"] for r in images if r["image_uid"] in search}
    check("PARENTS", "عداد منظر السيارة وأهلية البحث في الإعلان يطابقان الصور",
          all(int(r["n_overview_images"]) == overview_counts[r["listing_uid"]] and (r["usable_visual_search"] == "True") == (r["listing_uid"] in search_parents) for r in listings))
    file_tasks = {"recognition": "recognition_images", "make_recognition": "make_recognition_images", "color": "color_images", "body_type": "body_images", "visual_search": "visual_search_images"}
    check("TASK-FILES", "كل مجموعة مهمة محفوظة تطابق الصفوف المؤهلة كاملة",
          all(read_csv(processed / (name + ".csv")) == [r for r in images if r["usable_" + task] == "True"] for task, name in file_tasks.items()))
    check("PRICE-EXCLUSIONS", "حالات الممشى المستبعدة من السعر محفوظة كاملة دون فقد الإعلان",
          read_csv(processed / "price_exclusions.csv") == [r for r in listings if r["condition"] == "used" and r["usable_price"] != "True"])
    check("COUNTS", "ملخص المحتوى وأعداد كل مهمة مطابقان للجداول",
          summary["content_review"]["reviewed_images"] == len(images) and summary["content_review"]["kinds"] == dict(Counter(r["content_kind"] for r in images))
          and all(summary["task_images"][task] == sum(r["usable_" + task] == "True" for r in images) for task in file_tasks))
    pairs = read_csv(review / "expanded_duplicates/priority_pairs.csv")
    judgments = [r for p in (review / "expanded_duplicates/decisions").glob("*.csv") for r in read_csv(p)]
    check("SIMILARITY-REVIEW", "كل مرشح تشابه أولوية له حكم يدوي؛ التشابه وحده لا يثبت التكرار",
          len(judgments) == len(pairs) and {r["review_index"] for r in judgments} == {r["review_index"] for r in pairs}
          and all(r["review_judgment"] == "not_same_photo" for r in judgments), {"reviewed_pairs": len(judgments)})
    if supplement:
        from day20_supplemental_validate import supplemental_checks
        checks.extend(supplemental_checks(processed, listings, images, summary, supplement))
    if summary.get("user_issues"):
        from day21_issue_validate import issue_checks
        checks.extend(issue_checks(processed, listings, images, schema, summary))
    return checks
