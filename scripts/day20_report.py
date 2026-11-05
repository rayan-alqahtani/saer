"""توليد تقرير اليوم بالعربية من النتائج المحفوظة والمجربة فقط."""
from __future__ import annotations
from collections import Counter
import importlib.metadata
import json
import platform
import sys
from day20_common import *

AR_SOURCE = {"external_group_a": "الياباني", "external_group_b": "الأمريكي والصيني", "external_group_c": "الكوري والألماني"}
AR_REASON = {
    "no_verified_usable_image": "تعذر توفير صورة سليمة موثقة",
    "ambiguous_or_missing_model": "موديل عام أو غير محسوم من المصدر",
    "unverified_model_from_url": "اسم موديل من الرابط لم تؤكده بيانات الإعلان",
    "missing_or_invalid_price": "سعر مفقود أو غير صالح",
    "missing_or_invalid_used_mileage": "ممشى المستعمل مفقود أو غير صالح",
    "missing_city": "مدينة غير متاحة", "missing_transmission": "ناقل الحركة غير متاح",
    "missing_type_fuel": "نوع الوقود غير متاح", "historical_identity_mismatch": "بيانات مخبأة لإعلان آخر",
    "source_identity_conflict_make": "اختلاف الماركة بين السجل والصفحة الحالية",
    "source_identity_conflict_year": "اختلاف سنة الصنع بين السجل والصفحة الحالية",
    "source_identity_conflict_condition": "اختلاف حالة السيارة بين السجل والصفحة الحالية",
    "year_out_of_scope": "سنة خارج النطاق", "invalid_year": "سنة غير صالحة",
    "invalid_condition": "حالة سيارة غير صالحة", "invalid_source_identity": "هوية المصدر غير صالحة",
}

def table(headers, rows):
    rows = [[str(v).replace("|", "/").replace("\n", " ") for v in row] for row in rows]
    return "\n".join(["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |", *["| " + " | ".join(row) + " |" for row in rows]])


def save_rtl_bytes(path, content):
    body = content.decode("utf-8")
    if not body.lstrip().startswith('<div dir="rtl">'):
        body = '<div dir="rtl">\n\n' + body.rstrip() + '\n\n</div>\n'
    save_bytes(path, body.encode("utf-8"))

def report():
    quality = load_json(PROCESSED / "quality_report.json")
    if quality and quality.get("content_review"):
        from day20_content_report import report as reviewed_report
        return reviewed_report()
    publication_note = ""
    if PROCESSED != ROOT / "data/processed":
        publication_note = "**حالة الحفظ: هذه نتائج النسخة المراجعة المرحلة في `" + relative(PROCESSED) + "`. لم تحفظ بعد مكان ملفات data/processed والملفات المهمة. نتائج الاختبارات أدناه تخص النسخة المرحلة حتى اكتمال فحص النشر.**\n\n"
    validation = load_json(AUDIT / "validation.json")
    unit = load_json(AUDIT / "unit_tests.json")
    if not quality or not validation or not validation["passed"] or not unit or not unit["passed"]:
        raise SystemExit("لا تقرير إنجاز قبل نجاح التحقق والاختبارات.")
    if validation.get("processed_data_path") != relative(PROCESSED):
        raise SystemExit("نتائج التحقق تخص نسخة بيانات أخرى؛ تحقق من النسخة المطلوبة أولًا.")
    before = load_json(AUDIT / "before.json")
    gates = load_json(AUDIT / "source_gates.json")
    recovery = load_json(RECOVERY / "manifest.json")
    listings = read_csv(PROCESSED / "listings.csv")
    images = read_csv(PROCESSED / "images.csv")
    observed_recognition_labels = len({r["rec_label"] for r in images if r["usable_recognition"] == "True"})
    quarantine = read_csv(PROCESSED / "quarantine_listings.csv")
    issues = read_csv(AUDIT / "issues.csv")
    changes = read_csv(AUDIT / "changes.csv")
    raw_files = load_json(AUDIT / "raw_manifest.json")["total_files"]
    all_rows = {source: read_csv(STAGING / source / "listings_all.csv") for source in SOURCES}
    source_table = table(["المجموعة", "إعلانات الأصل", "الإعلانات المعتمدة", "الإعلانات المعزولة", "الصور المعتمدة", "بوابة المصدر"], [[AR_SOURCE[s], g["original_listings"], g["accepted_listings"], g["quarantined_listings"], g["accepted_images"], "ناجحة" if g["passed"] else "فاشلة"] for s, g in gates.items()])
    fixes = []
    for source in SOURCES:
        original = read_csv(RAW / source / SOURCES[source])
        changed_prices = {r["listing_uid"] for r in changes if r["source_dataset"] == source and r["field"] == "sar_price" and not text(r["before"]) and number(r["after"]) is not None}
        fixes.append([AR_SOURCE[source], before["sources"][source]["listings_without_images"], sum(int(r["n_images"]) == 0 for r in all_rows[source]), sum(not text(r["sar_price"]) for r in original), len(changed_prices), sum(not text(r["sar_price"]) for r in all_rows[source])])
    fixes_table = table(["المجموعة", "بلا صور قبل", "بلا صور بعد، معزولة", "بلا سعر قبل", "أسعار استرجعت", "بلا سعر بعد، معزولة"], fixes)
    reason_counter = Counter(code for r in quarantine for code in r["quarantine_reasons"].split(";") if code)
    reasons_table = table(["سبب العزل", "عدد الإعلانات"], [[AR_REASON.get(reason, reason), count] for reason, count in reason_counter.most_common()])
    optional = table(["الحقل", "متوفر في المعتمد", "غير متوفر في المعتمد"], [[f, sum(bool(text(r[f])) for r in listings), sum(not text(r[f]) for r in listings)] for f in ["trim", "color", "body_type", "listing_date"]])
    source_statuses = Counter(str(v["status"]) for v in recovery["pages"].values())
    issue_actions = Counter(r["action"] for r in issues)
    issue_categories = Counter((r["issue_code"], r["action"]) for r in issues)
    pair_counts = quality["duplicate_pairs"]
    external_review = load_json(AUDIT / "external_analysis_review.json", {})
    final_review = load_json(AUDIT / "external_analysis_final_review.json", {})
    review_section = ""
    if final_review.get("passed"):
        if final_review.get("processed_data_path") != relative(PROCESSED):
            raise ValueError("مراجعة التحليل تخص نسخة أخرى؛ أعد فحص النسخة المطلوبة قبل التقرير")
        review_section = "## نتيجة ملاحظات التحليل بعد الاستكمال\n\n"
        review_section += "راجعت الملاحظات الـ25 في ملف المستخدم مجددًا على النسخة الحالية بعد الاستكمال. الأعداد والإجراءات التالية من فحوص هذه النسخة. التفاصيل في `data/audit/day20/external_analysis_final_review.json`.\n\n"
        review_section += table(["#", "الملاحظة", "النتيجة", "الإجراء أو الحد"], [[r["number"], r["problem"], r["verdict_ar"], r["action_ar"]] for r in final_review["findings"]]) + "\n\n"
    elif external_review.get("passed"):
        review_section = "## مراجعة التحليل الإحصائي المرفق\n\n"
        review_section += "في المراجعة الأولى للتحليل المرفق، راجعت الملاحظات الـ25 وطابقت الأسعار المسترجعة الـ295 مع هويات صفحاتها وأسعارها وبصماتها. كان مصدر أسعار الجديد 295 مسترجعة و143 من الأصل. جدول الملاحظات التالي يحفظ نتائج تلك المرحلة؛ نتائج الاستكمال اللاحق موضحة بعدها وفي إحصاءات النسخة الحالية.\n\n"
        review_section += table(["#", "الملاحظة", "النتيجة", "الإجراء أو الحد"], [[r["number"], r["problem"], r["verdict_ar"], r["action_ar"]] for r in external_review["findings"]])
        review_section += "\n\nعلامات المراجعة في data_quality_flags لا تدخل ميزات النموذج ولا تعني خطأ مؤكدًا أو احتيالًا. المؤشر السنوي تقريبي من سنة الجمع مع حد أدنى سنة، ولا يدعي معرفة تاريخ أول استخدام. قيم اللون والهيكل والتاريخ الناقصة تبقى غير متوفرة؛ لا مصدر محلي يملؤها، ولا يحول قاموس الموديلات إلى حقيقة عن الصورة. التفاصيل والأدلة في `data/audit/day20/external_analysis_review.json`.\n\n"
    if quality.get("enrichment", {}).get("complete"):
        enrichment = quality["enrichment"]
        review_section += "## الاستكمال الموسع من المصدر\n\n"
        review_section += f"فحصت جميع حالات قائمة الاستكمال وعددها {enrichment['queue_size']} بعد موافقة المستخدم. نتائج المصدر: {json.dumps(enrichment['status_counts'], ensure_ascii=False)}. استكملت الحقول والوسوم من الإعلان المطابق فقط، وحفظت HTML مضغوطًا مع بصمته وJSON المستخرج منه. التصنيفات المشتقة من المعرض مرتبطة بالصورة الفعلية بالمسار والاسم؛ لا تخمين لخارجية الصورة من ترتيبها.\n\n"
        review_section += f"أهلية تصنيف الماركة: {quality.get('make_recognition_ready_images', 0)} صورة ضمن {quality.get('supported_make_labels', 0)} ماركات مدعومة، مع حفظ هويات الموديلات النادرة كما هي. حالات تأكيد الممشى لدى القيم المعلّمة: {json.dumps(quality.get('mileage_source_statuses', {}), ensure_ascii=False)}. القيم التاريخية لا تستبدل بقيمة من وقت آخر دون تمييز؛ current_source_mileage وsource_verified_at يسجلان المراجعة منفصلة.\n\n"
        review_section += "الفئة التسويقية trim منفصلة عن مستوى التجهيز؛ لم تستعمل كلمة Standard أو Half Full لتعبئة فئة لم يذكرها المصدر. النموذج البصري لا يشمل كل الموديلات لمجرد وجودها في البيانات. الحالات التي تعذر استرجاع مصدرها أو صورها لها مصير صريح ولا توصف بأنها أصلحت.\n\n"
        source_review = load_json(AUDIT / "enrichment_review.json", {})
        if source_review.get("passed"):
            review_section += table(["الحقل", "المتوفر قبل الاستكمال", "المتوفر بعد الاستكمال", "المتبقي غير متوفر"], [[field, counts["before_known"], counts["after_known"], counts["after_missing"]] for field, counts in source_review["optional_coverage"].items()]) + "\n\n"
            review_section += f"جميع تغييرات الحقول وعددها {source_review['source_verified_change_count']} مطابقة لمراجعها، وحفظت أسعار وممشى السجلات التاريخية المعتمدة. استنقذت {len(source_review['rescued_listing_uids'])} حالة من العزل؛ الحالات التي استدعى الفحص عزلها بعد المراجعة {len(source_review['new_quarantine_uids'])}. لا فقد لأي سجل أصلي.\n\n"
            original_check = source_review.get("original_snapshot_check", {})
            if original_check:
                original_counts = original_check["compared_values"]
                review_section += f"فحص نقل اللقطة الأصلية: طابقت {original_counts.get('sar_price', 0)} قيمة سعر و{original_counts.get('km_mileage', 0)} قيمة ممشى بين CSV وكاش المصدر المطابق للهوية، دون اختلاف. هذا يثبت مطابقة النقل حيث تتوفر القيمتان، ولا يدعي فحص العداد أو جميع القيم التي تعذر مقارنتها.\n\n"
    base = f"""<div dir="rtl">

# تدقيق البيانات وإنجاز 20 أكتوبر 2026

{publication_note}
## نتيجة اليوم

اكتمل الجرد والاسترجاع الممكن والتنظيف مستقلًا لكل مصدر، ثم جرى الدمج الأولي بعد نجاح بوابات المصادر الثلاثة. نجح {validation['passed_count']} فحصًا مستقلًا من {validation['total_checks']}، ونجح {unit['tests_run']} اختبارًا آليًا دون تخطٍّ. لم يبدأ تدريب النماذج أو بناء الواجهة.

المدخلات {quality['raw_listings']:,} إعلانًا و{quality['raw_images']:,} صورة. المخرجات المعتمدة {quality['clean_listings']:,} إعلانًا و{quality['clean_images']:,} صورة. عزل {quality['quarantined_listings']:,} إعلانًا مع السبب والدليل؛ لم تحذف هذه السجلات من الأصل ولم تستبدل ببيانات تخمينية.

{source_table}

## قبل الاسترجاع وبعده

{fixes_table}

في جولة الإصلاح الأولى فحصت {len(recovery['pages'])} صفحة ضمن قائمة المشاكل المحددة؛ توفرت بيانات مطابقة لهوية الإعلان في {source_statuses.get('ok', 0)} صفحة. نتائج جولة الاستكمال الموسعة مبيّنة أدناه بصورة منفصلة. حفظت الصفحات والنتائج وتواريخها وبصماتها. إجمالي الصور السليمة المسترجعة في مخرجات اليوم {quality['recovered_images']} صورة. الإعلانات التي تحولت إلى قوائم سيارات أخرى لم تستعر منها أي حقول أو صور.

اكتملت أيضًا محاولات روابط المعرض الأصلية والروابط التاريخية التي لم تتوفر صورها، دون إضافة صور أخرى. وأكد تحقق إضافي لمقاسين من صورة واحدة تعيد خطأ خادم 500 استمرار العطل، ولم يغير الجداول المعتمدة؛ دليله `data/recovery/transient_checks.json`.

## الأسباب الجذرية والإصلاح

1. **أسعار وحقول موجودة لم يلتقطها السكربت القديم:** كان يبحث عن أول ظهور للمعرّف ضمن نافذة نصية، وقد يكون في قائمة ألوان الإعلان. أصبح الاستخراج من كائن `postDetails`، مع مطابقة معرّفات `details` و`g4Data` و`analytics` مع الإعلان المطلوب، ومطابقة السعر النقدي مع السعر شامل الضريبة عندما يتوفر.
2. **صور HEIC مستبعدة من تعبير البحث القديم:** يقرأ الإصلاح معرض الإعلان مباشرة دون قصر امتداد URL، ويتحقق من تنسيق الملف الفعلي. تحفظ البايتات المسترجعة كما وردت دون تحويل صور تخميني.
3. **معرّفات تاريخية لا تطابق البيانات:** اكتشف سجلان من الكوري والألماني يحملان بيانات لإعلان آخر. عولج سبب الهوية صراحة، والحالات التي تعذر إثبات هويتها من الصفحة الصحيحة معزولة.
4. **أسماء موديلات وحقول فارغة أو غير متسقة:** استكملت من الحقل الخام أو المخبأ المطابق للهوية أو الصفحة الصحيحة، مع توحيد تنسيق موثق. في BMW أعيد تركيب الموديل من عائلة X ورقم صريح في بداية حقل الفئة، دون استنتاج من صورة أو محرك. أسماء العائلات التي لا تحسمها هذه الأدلة لا تدخل المعتمد.
5. **مسارات قديمة وامتدادات غير مطابقة:** ربط الصور بالملفات الموجودة فعليًا وبمعرّف الإعلان، وأصبحت المسارات نسبية إلى جذر المشروع. كل مسار معتمد موجود وهاشه مطابق.
6. **صور مكررة أو متشابهة:** وجد {pair_counts.get('exact_sha256', 0)} زوجًا مرجعيًا للتكرار الثنائي و{pair_counts.get('near_duplicate_suspected', 0)} زوجًا بصريًا مشتبهًا. أزيل تكرار الصورة داخل الإعلان من النسخة المعتمدة فقط، وجمعت الإعلانات المرتبطة احترازيًا في قسم واحد. التشابه البصري اشتباه قابل للتفسير، وليس إثباتًا بأن السيارتين واحدة.
7. **صور ذات وسوم متعارضة:** عزلت الصور التي لا يمكن اعتماد نسبتها للفئة بأمان. كل استبعاد مسجل، ولا حذف من القرص الخام.
8. **الأسعار الشاذة:** سجلت العلامات الوصفية في ملف مستقل. لم تعوض الأسعار أو تحذف لمجرد خروجها عن التوزيع، ولم تستخدم هذه العلامات كسمات للنموذج.

## العزل والبيانات غير القابلة للاسترجاع

{reasons_table}

قد يجتمع أكثر من سبب في السجل الواحد؛ أعداد الأسباب ليست مجموع الإعلانات المعزولة. البيانات المستبعدة وأسبابها في `data/processed/quarantine_listings.csv`، والصور المستبعدة في `data/processed/image_exclusions.csv`. مصير كل رابط تاريخي مفقود موثق في سجل المشاكل: استرجاع مطابق أو عزل للرابط القديم، دون تمريره إلى الواجهة كمسار صالح.

## الحقول الاختيارية وحدود الوسوم

{optional}

عدم توفر الحقول الاختيارية موثق ولا يملأ بتخمين. ممشى السيارات الجديدة المفقود لم يحول إلى صفر. وسوم اللون والهيكل مأخوذة من بيانات المصدر المرتبطة بالإعلان؛ تغطيتها الفعلية في الجدول، ولا تدعي هذه المرحلة دقة نموذج لون أو هيكل مدرب.

يوجد {quality['verified_exterior_images']} صورة خارجية أكدها معرض المصدر، و{quality['recognition_ready_images']} صورة مؤهلة مبدئيًا للتعرف موزعة على {observed_recognition_labels} فئات مدعومة فعليًا في الصور. يوجد {quality.get('listing_supported_model_labels', 0)} فئة بدعم كاف في جدول الإعلانات؛ هذا لا يثبت دعم صورها. الأهلية البصرية تتطلب 12 مجموعة إعلان مستقلة على الأقل بصور خارجية فريدة، منها 8 للتدريب وواحدة على الأقل لكل من التحقق والاختبار. تحفظ الفئات الأخرى بهويتها الأصلية مع usable_recognition=False وسبب واضح، ولا تدمج عشوائيًا في فئة أخرى. الصور غير الموسومة تنتظر الفلترة؛ لا تفترض خارجية الصورة من ترتيبها. لم يدرب أو يقيم نموذج تعرف بعد.

## منع التسرب وتكرارية المعالجة

بذرة التقسيم 42. التقسيم على مجموعة الإعلان وربط الصور المطابقة والمشتبه بها والتطابقات الوصفية، لا على الصورة منفردة. التقسيم الفعلي: {json.dumps(quality['split']['actual_listings'], ensure_ascii=False)}. لا إعلان أو هاش صورة أو مجموعة تكرار تعبر التدريب والتحقق والاختبار. فهارس التقسيم وتفاصيل الطريقة محفوظة في `data/processed/splits/`.

مخطط السمات يحظر المعرّفات والمسارات والمصدر والهدف من مدخلات نموذج السعر. مجموعة السعر المهيأة تحتوي {quality['price_rows']} إعلانًا مستعملًا؛ تقدير الجديد يحتاج تقييمًا منفصلًا قبل عرضه كخدمة جاهزة.

## حفظ الأصل وحقوق المصدر

حفظت نسخة خام من {raw_files:,} ملفًا، وتحققت من SHA-256 للأصل والنسخة مجددًا بعد المعالجة. لم يعدل أي ملف في المجلدات الأصلية. كل استرجاع في `data/recovery/` مع التاريخ والدليل؛ كل تغيير حقل في `data/audit/day20/changes.csv`. تحفظ إصدارات النتائج السابقة تلقائيًا تحت `data/history/` عند اختلافها.

احترم الاسترجاع robots.txt وحدًا قدره طلب واحد كل 2.1 ثانية لكل مضيف، دون تسجيل دخول أو API أو تجاوز حظر. بعض الصور تعيد 404، وتبقى حالات عدم التوفر موثقة. حقل إذن الاستخدام في البيانات ما زال يشير إلى انتظار الإذن المكتوب؛ مخرجات اليوم محلية، ولم ترفع بيانات أو صور إلى GitHub.

تواريخ الجمع الأصلية لقطات بين 13 و18 أكتوبر، وإصلاحات المصدر مؤرخة في 20 أكتوبر. هذه ليست سلسلة زمنية متجانسة لتحليل اتجاه السوق، ولا دليلًا على توفر الإعلانات حيًا أو أسعار بيع منجزة.

نطاق تواريخ الإدراج التي وفرها المصدر في المعتمد: {json.dumps(quality.get('listing_date_range'), ensure_ascii=False)}. تاريخ الإدراج منفصل عن تاريخ الجمع ووقت التحقق؛ لا يفترض أن كل الإعلانات نُشرت حديثًا.

## الأدلة والمخرجات

- `data/audit/day20/before.json`: الجرد الأصلي وتقارير المصادر والسكربتات والأرشيفات.
- `data/audit/day20/issues.csv`: سجل المشاكل كاملًا؛ إصلاحات {issue_actions['fixed']}، حالات عزل {issue_actions['quarantined']}، وحدود موثقة {issue_actions['documented']}. هذه أعداد قضايا وقد يخص السجل الواحد أكثر من قضية.
- `data/audit/day20/changes.csv`: قبل وبعد ودليل كل تغيير.
- `data/staging/day20/<source>/gate.json`: فحوص المصدر قبل الدمج، مع جداول المعتمد والمعزول.
- `data/processed/listings.csv` و`images.csv`: الجداول الأولية الموحدة المعتمدة.
- `data/processed/price_dataset.csv` و`recognition_images.csv` و`class_support.csv`: مجموعات المهام والدعم الفعلي للفئات.
- `data/processed/schema.json` و`quality_report.json`: المخطط، القيود، الملخص والمقاييس الفعلية.
- `data/audit/day20/validation.json` و`unit_tests.json`: نتائج الفحوص والاختبارات.
- `docs/SCRIPT_REVIEW.md` و`docs/TEST_REPORT.md`: مراجعة السكربتات والتحقق.

{review_section}## المتبقي وفق الخطة

مهمة 20 أكتوبر منجزة مع عزل معلن للحالات التي تعذر إصلاحها، دون ادعاء استرجاع ما ليس متاحًا. يوم 21 مخصص لخط أساس السعر ونموذج CPU، وفلترة الصور وembeddings وفهرس التشابه. يوم 22 لاجتياز اختبار حفظ وتحميل النماذج والاستدلال وتوثيق حدود التعرف. لا نماذج جديدة مدربة ضمن مخرجات هذا اليوم. صورة الداشبورد المرجعية غير موجودة في الحزمة؛ تسجل لمهمة الواجهة في موعدها ولا تعطل عمل البيانات.

</div>
"""
    save_rtl_bytes(ROOT / "docs/DATA_AUDIT.md", base.encode("utf-8"))
    tests = table(["الفحص", "الوصف", "النتيجة"], [[r["id"], r["description_ar"], "ناجح" if r["passed"] else "فاشل"] for r in validation["checks"]])
    save_rtl_bytes(ROOT / "docs/TEST_REPORT.md", (f"# تقرير الاختبارات — 20 أكتوبر 2026\n\n{tests}\n\nنجحت {unit['tests_run']} اختبارات آلية، دون أخطاء أو تخطٍّ. اختبار Gitignore تحقق من {unit['gitignore']['checked_paths']} مسارًا في مستودع مؤقت منفصل، دون staging أو commit أو push في المشروع.\n\nالأمر المجرب: `powershell -ExecutionPolicy Bypass -File scripts/run_day20.ps1 -Stage Tests`، والتحقق: المرحلة `Validate`.\n\nاختبارات تعلم الآلة والخادم والواجهة خارج مهمة اليوم ولم تنفذ.\n").encode("utf-8"))
    save_rtl_bytes(ROOT / "docs/SCRIPT_REVIEW.md", """# مراجعة السكربتات السابقة

جرت مراجعة ملفات المصادر قبل بناء خط التنظيف. حفظت نسخها وهاشاتها، ولم تعدل الأصول.

| الملف السابق | نتيجة المراجعة والإجراء |
| --- | --- |
| collect_from_sitemap.py في المصادر | أسلوب نافذة أول post_id قد يلتقط بيانات اللون أو إعلانًا مختلفًا. يعتمد الإصلاح postDetails كاملًا ويفرض ثلاثة تطابقات للهوية. |
| download_images.py | يدعم استئناف التنزيل والتحقق من نوع الملف؛ استفيد من احترام robots والتوقف عند الحظر. حصر الروابط القديمة في jpg/png/webp كان يفقد HEIC. لم يشغل السكربت على الأصل كي لا يستبدل ملفاته. |
| verify_images.py | تقارير تحقق قديمة مفيدة، لكنها ليست بديلًا عن جرد فعلي. أعيد فحص جميع الملفات وبصماتها، وربطت المسارات الحقيقية بدل امتداد الكتالوج المفترض. |
| 01_prepare_data.py | أساس مفيد للجودة وتجميع التكرار. يحتاج أصل مسارات مستقل لكل مصدر، مصدرًا لكل حقل، واستبعاد أسماء الموديلات العامة غير المثبتة. لم تعتمد تحويلات تدمج موديلين مختلفين تلقائيًا. |
| 02_price_model.py | توجد OOF وCV على KG فقط، دون نموذج نهائي محفوظ. المعالج خفيف ومفيد ليوم 21؛ لا تعتمد مقاييسه القديمة على أنها تقييم للمصادر الموحدة. ظهرت حالتا هوية خاطئة في مدخلات تاريخية. |
| 03_train_recognizer.py | ملف تدريب أولي يطلب أوزانًا ومكتبات غير مجهزة هنا؛ لم يثبت تشغيله سابقًا. الإعداد الافتراضي للدفعة والتدريب لا يعتمد على CPU قبل تجربة صغيرة. تدريب الماركة ثم الفئات المدعومة هو ترتيب الخطة. |
| 04_visual_search.py | توجد آلية بحث وتجميع حسب الإعلان. مسار CLIP يحتاج تنزيل أوزان وتجربة فعلية يوم 21. لا ملف embeddings أو فهرس محفوظ جاهز ضمن الأصول. |
| make_data_guide.py في USA | يحيل إلى project_engine.runner، والحزمة غير مرفقة في الملفات الحالية. لا يعد نقطة تشغيل جاهزة رغم README القديم. |

الفحص النحوي لكل Python وCRC للأرشيفات مسجل في before.json. لم تنشأ أرقام تدريب أو تنبؤات جديدة ولم تعدل نتائج السعر القديمة.
""".encode("utf-8"))
    save_rtl_bytes(ROOT / "docs/ISSUE_REGISTER.md", ("# سجل المشاكل — 20 أكتوبر 2026\n\nالسجل التفصيلي القابل للتصفية في `data/audit/day20/issues.csv`. كل صف يحمل معرّفًا ومصدرًا وكيانًا وحقلًا والقيمة الأصلية والنتيجة ودليل المعالجة.\n\n" + table(["رمز المشكلة", "الإجراء", "عدد الحالات"], [[code, {"fixed": "أصلحت", "quarantined": "عزلت", "documented": "وثقت وفق حدود العقد"}[action], count] for (code, action), count in sorted(issue_categories.items())]) + "\n\nكل حالة لها مصير موثق. القيم غير المعتادة المحتفظ بها تحمل علامات مراجعة، ولا يعني اعتماد المصدر إثبات خلو كل قيمة من الخطأ. العزل لا يعني حذف الأصل أو الادعاء باسترجاع الناقص.\n").encode("utf-8"))
    env = {"python": sys.version, "platform": platform.platform(), "packages": {m: importlib.metadata.version(m) for m in ["numpy", "Pillow", "openpyxl"]}, "seed": 42, "pipeline_script_sha256": {p.name: sha256(p) for p in (ROOT / "scripts").glob("day20_*.py")}}
    save_json(AUDIT / "environment.json", env)
    # احتفظ بالجدول المرحلي الأصلي، وحدث المنجز الحقيقي فقط.
    progress = (ROOT / "PROGRESS.md").read_text(encoding="utf-8")
    progress = progress.replace("**الحالة:** تنفيذ يوم 20 أكتوبر جارٍ. اكتمل الجرد والنسخ الخام والتحقق الأولي، وجارٍ استرجاع النواقص قبل التنظيف النهائي والدمج. لم يبدأ التدريب أو بناء الواجهة.", "**الحالة:** اكتمل عمل 20 أكتوبر: جرد وإصلاح وتنظيف المصادر، ثم الدمج الأولي بعد اجتياز بوابات الجودة. الحالات المتعذرة معزولة وموثقة. النماذج والواجهة لم تبدأ.")
    progress = progress.replace("| M1 — تنظيف ودمج البيانات | حتى 22 أكتوبر | جارٍ، دون دمج حتى اجتياز بوابات المصادر | `data/audit/day20/before.json` و`data/recovery/manifest.json` |", "| M1 — تنظيف ودمج البيانات | حتى 22 أكتوبر | مكتمل نطاق يوم 20 أكتوبر مع عزل موثق | `docs/DATA_AUDIT.md` و`data/audit/day20/validation.json` |")
    history = ""
    if "## آخر جلسة" in progress:
        old_session = progress.split("## آخر جلسة", 1)[1]
        if "\n## " in old_session:
            history = "\n## " + old_session.split("\n## ", 1)[1]
    progress = progress.split("## آخر جلسة", 1)[0] + f"""## آخر جلسة
- **ما فُحص:** {quality['raw_listings']} إعلانًا و{quality['raw_images']} صورة أصلية، ملفات CSV وExcel والأرشيفات والسكربتات والنتائج السابقة. كل النسخ الخام متطابقة بالهاش والأصول سليمة بعد المعالجة.
- **ما أُنجز فعليًا:** استرجاع {quality['recovered_images']} صورة؛ تنظيف مستقل وبوابات ناجحة للمصادر الثلاثة؛ اعتماد {quality['clean_listings']} إعلانًا و{quality['clean_images']} صورة؛ عزل {quality['quarantined_listings']} إعلانًا بأسباب وأدلة؛ فهارس تقسيم بseed=42 دون تسرب؛ مخطط ومجموعات المهام وتقارير عربية.
- **الاختبارات والنتائج:** {validation['passed_count']}/{validation['total_checks']} فحص جودة مستقل ناجح؛ {unit['tests_run']} اختبارًا آليًا ناجحًا دون تخطٍّ؛ حماية Gitignore لـ{unit['gitignore']['checked_paths']} مسارًا مجربة. لا اختبارات تدريب أو API أو واجهة ضمن نطاق اليوم.
- **الأوامر المجربة:** مراحل `scripts/run_day20.ps1`: Inventory ثم Recover ثم Clean ثم Validate ثم Tests ثم Report. الاسترجاع وحده يحتاج اتصالًا عامًا بالمصدر.
- **الملفات المتغيرة:** وثائق MD في الجذر؛ `.gitignore`؛ `requirements-data.txt`؛ `README.md`؛ ملفات scripts وtests الجديدة؛ تقارير docs؛ ملفات data الجديدة فقط. لم تعدل مجلدات الأصول ولم يحدث Git staging أو رفع.
- **العوائق والحدود:** التفاصيل في DATA_AUDIT.md. الإعلانات غير المتاحة أو ذات الهوية/الموديل غير المحسوم معزولة. وسوم اللون والهيكل والخارج/الداخل جزئية وموثقة. صورة تصميم الداشبورد غير مرفقة.
- **تحقق إضافي:** إعادة خطأ الخادم لصورة واحدة بمقاسين أكدت استمرار تعذرها، دون تغيير الجداول المعتمدة. الدليل `data/recovery/transient_checks.json`.
- **الموافقة:** لا توجد موافقة إضافية مطلوبة لإغلاق عمل يوم 20 أكتوبر.
- **الخطوة التالية حسب ROADMAP:** 21 أكتوبر: خط أساس السعر ونموذج CPU وembeddings وفهرس صور؛ ثم اختبارات بوابة النماذج يوم 22. لا تبدأ الواجهة قبل الأساس القابل للاختبار.
"""
    save_bytes(ROOT / "PROGRESS.md", (progress + history).encode("utf-8"))
    print(json.dumps({"report": "docs/DATA_AUDIT.md", "day20_status": quality["day20_status"], "clean_listings": quality["clean_listings"], "quarantined": quality["quarantined_listings"]}, ensure_ascii=False), flush=True)

if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    report()
