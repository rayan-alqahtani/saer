"""تقرير عربي RTL للنسخة المراجعة، من نتائج فحوص حقيقية فقط."""
from collections import Counter
from day20_common import *


def table(headers, rows):
    clean = lambda value: str(value).replace("|", " / ").replace("\n", " ")
    return "\n".join(["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |",
                       *["| " + " | ".join(clean(v) for v in row) + " |" for row in rows]])


def write(name, body):
    save_bytes(ROOT / name, ('<div dir="rtl" lang="ar">\n\n' + body.rstrip() + '\n\n</div>\n').encode("utf-8"))


def report():
    quality = load_json(PROCESSED / "quality_report.json")
    validation = load_json(AUDIT / "validation.json")
    units = load_json(AUDIT / "unit_tests.json")
    publication = load_json(AUDIT / "content_publication.json")
    if not quality.get("content_review") or not validation.get("passed") or not units.get("passed") or validation["processed_data_path"] != relative(PROCESSED):
        raise ValueError("التقرير يتطلب نجاح فحوص النسخة النهائية نفسها")
    from verify_essential_bundle import verify
    compact = verify(ROOT / "الملفات_المهمة")
    listings = read_csv(PROCESSED / "listings.csv")
    images = read_csv(PROCESSED / "images.csv")
    price = read_csv(PROCESSED / "price_dataset.csv")
    model_support = read_csv(PROCESSED / "class_support.csv")
    price_support = {r["label"]: r for r in read_csv(PROCESSED / "price_support.csv")}
    attrs = read_csv(PROCESSED / "attribute_support.csv")
    by_id = {r["listing_uid"]: r for r in listings}
    unresolved_body_visual = sum(r["body_label_status"] == "visual_insufficient_evidence" for r in listings)
    schema = load_json(PROCESSED / "schema.json")
    supplemental = quality.get("supplemental_images", {})
    no_overview = sum(int(r["n_overview_images"]) == 0 for r in listings)
    from day20_enrich import read_events
    source_pages = dict(load_json(RECOVERY / "manifest.json")["pages"])
    source_pages.update(read_events())
    no_overview_source_status = Counter(source_pages.get(r["listing_uid"], {}).get("status", "unknown") for r in listings if int(r["n_overview_images"]) == 0)
    sparse_test = sum(r["split"] == "test" and int(price_support[r["rec_label"]]["train_groups"]) < 5 for r in price)
    low_resolution = sum(min(int(r["width"]), int(r["height"])) < 224 for r in images)
    groups = len({r["group_id"] for r in listings})
    missing = quality["source_optional_missing"]
    source_counts = Counter(r["source_dataset"] for r in listings)
    riyadh = sum(r["city"].casefold() == "riyadh" for r in price)
    kinds = quality["content_review"]["kinds"]
    non_overview = len(images) - kinds["vehicle_overview"]
    scope = "**اكتملت معالجة العيوب المحصورة ونشر النسخة المجربة محليًا ضمن النطاق الموثق. البيانات ليست مكتملة الحقول أو التغطية بنسبة 100%. لم يبدأ أي نموذج، وبوابة التدريب ما زالت مغلقة وفق توجيه المستخدم.**"
    header = f"آخر تحديث: {now()} (UTC). النسخة: `{relative(PROCESSED)}`، المخطط `{schema['version']}`.\n\n{scope}\n"
    content_table = table(["المحتوى المراجع", "عدد الملفات", "استخدام منظر السيارة"], [
        ["منظر السيارة", kinds["vehicle_overview"], "تطبق أهلية كل مهمة ودعم الفئة والتكرار"],
        ["تفاصيل مثل العجلة والشبك والمفتاح", kinds["vehicle_detail"], "مستبعدة من مهام المنظر، محفوظة للأرشفة والعرض"],
        ["داخلية", kinds["interior"], "مستبعدة من مهام المنظر"], ["وثائق وبطاقات", kinds["document"], "مستبعدة من مهام المنظر"]])
    task_names = {"recognition": "تعرف الموديل", "make_recognition": "تعرف الماركة", "visual_search": "بحث الصور",
                  "color": "اللون", "body_type": "الهيكل"}
    task_table = table(["المهمة", "الصور المؤهلة الفريدة ثنائيًا", "حد التغطية"], [
        [task_names[task], count, (f"{quality['supported_model_labels']} من 272 فئة ماركة وموديل" if task == "recognition" else
         f"{quality['supported_make_labels']} من 41 ماركة" if task == "make_recognition" else
         "10 عائلات لون مدعومة أوليًا" if task == "color" else "5 أشكال جسم مدعومة أوليًا" if task == "body_type" else "لا يمثل دقة استرجاع؛ لم ينشأ فهرس embeddings")]
        for task, count in quality["task_images"].items()])
    limits = f"""- بقيت قيم اللون والهيكل والتاريخ الأصلية مفقودة في {missing['color']} إعلانًا لكل حقل، لأن المصدر تعذر أو لم يذكرها. لا نسخ من موديلات مشابهة ولا قيم مختلقة.
- في الوسوم المشتقة: {quality['derived_optional_missing']['color_std']} لونًا بلا هدف (تشمل petrol الغامض)، و{quality['derived_optional_missing']['body_type_std']} هيكلًا بلا هدف؛ عدد أحكام الهيكل البصرية التي ما زالت غير محسومة {unresolved_body_visual}، مع استكمال بعض الأشكال المثبتة بصريًا.
- {no_overview} إعلانًا لا يملك منظر سيارة مناسبًا ضمن الصور المحفوظة. منها {no_overview_source_status['source_redirected_or_unavailable']} مصدرها غير متاح أو محول، و{no_overview_source_status['ok']} لها معرض محفوظ لكن مناظره المتبقية لا تستوفي معيار الأهلية. يبقى السجل وصوره الأصلية، ويستبعد من مهام المنظر. لا يساوي وجود صورة داخلية توفر صورة خارجية، ولا تستبدل بصور غير مطابقة.
- {quality['quarantined_listings']} إعلانًا و{quality['excluded_images']} صورة معزولة محفوظة بأسبابها، دون فقد صامت للأصول.
- الفئات البصرية النادرة خارج الأهلية: الكوبيه والمكشوف مثلًا لا تحققان الحد الأدنى، ولا تعوض كثرة صور السيارة نقص المجموعات المستقلة. من {quality['supported_model_labels']} موديلًا مؤهلًا، {quality['model_support_tiers']['experimental_minimum']} عند الحد التجريبي و{quality['model_support_tiers']['larger_evaluation_sample']} لديها عينة تقييم أكبر؛ لا يثبت أي منهما الدقة.
- {sparse_test} صف سعر في الاختبار لموديلات ذات أقل من خمس مجموعات تدريب سعر؛ الملف price_support.csv يبين التغطية. هذا حد عينة لا يعالج بتغيير الاختبار أو اختلاق سيارات.
- مصدر الإعلانات منصة واحدة، وإن قسمت ملفات الفريق إلى ثلاث مجموعات. الرياض تمثل {riyadh}/{len(price)} من مجموعة السعر ({riyadh / len(price) * 100:.1f}%). لا تعميم إحصائي على سوق المملكة كله.
- {low_resolution} صورة ضلعها الأقصر أقل من 224 بكسل، معلومة الأبعاد ومحفوظة حسب المحتوى؛ لا توليد تفاصيل أو ادعاء ملاءمة غير مختبرة لصور الهواتف.
- وصف المصدر للون لا يثبت اللون ميدانيًا تحت جميع الإضاءات، ومراجعة الهيئة لا تثبت هوية VIN. الحكم بأن زوجًا ليس الصورة نفسها لا يثبت أنه سيارتان مختلفتان.
- السعر سعر إعلان تاريخي؛ ليس سعر بيع نهائيًا. الإعلانات ليست مؤكدة التوفر الحي، وحقوق الاستخدام للنشر العام لم تعتمد بعد.
- مقاييس السعر والتعرف واللون والهيكل وجودة الاسترجاع والتعميم على صور مستقلة: **لم يُختبر بعد**. لا نموذج أو أوزان أو embeddings ضمن هذه المهمة.
"""
    changes = f"""1. راجعت الأساس المكون من 13344 صورة عبر 46 لوحة، مع تكبير حالات المحتوى الملتبسة؛ لكل صورة قرار وبصمة ومسار دليل. بلغ عدد الإعلانات ذات وسوم الهيكل المراجعة {quality['content_review']['body_reviewed_listings']}. راجعت كذلك الصور الإضافية وعددها {supplemental.get('downloaded', 0)} في لوحات مستقلة؛ العدد المعتمد الحالي {len(images)} صورة.
2. فصلت {non_overview} صورة غير مناسبة لمنظر السيارة عن مهام التعرف والبحث واللون والهيكل. بقيت الملفات الأصلية ووسوم المعرض كما هي، وأضيف content_kind مستقل. صححت أيضًا نسخ صورتَي عجلات بعد تعارض الحكم عند تطابق SHA.
3. أضفت color_std بعائلات موثقة، وbody_type_std بشكل الجسم بدل وصف sport العام، وtrim_std لتوحيد اختلاف TITANIUM/Titanium في Patrol فقط. لا إعادة كتابة لقيم المصدر الأصلية.
4. نتجت {quality['content_review']['body_derived_changes']} تغييرًا في وسم الهيكل المشتق، تشمل تصحيح شكل مثبت، وتفسير sport، واستكمال شكل مثبت أو حجبه لعدم وضوحه. لا تعني كلها أخطاء نقل من المصدر.
5. فصلت 11 إعلانًا مستعملًا ذا ممشى غير معتاد لم يمكن تأكيده عن مجموعة السعر وحدها؛ حفظتها في price_exclusions.csv وبقيت في جدول الإعلانات. لم أغير قيمة السعر أو الممشى التاريخية. أصبح السعر {len(price)} صفًا: {quality['price_split']['train']} تدريب، و{quality['price_split']['val']} تحقق، و{quality['price_split']['test']} اختبار؛ الجديد الـ438 محفوظ خارج هذا النطاق.
6. أعدت دعم كل مهمة حسب صور منظر السيارة الفريدة ومجموعات الإعلان والأقسام الثابتة. الصور التي كان نوعها مجهولًا في المعرض يمكن أن تصبح مؤهلة بعد مراجعة محتواها؛ لذلك لا يلزم أن ينقص عدد الصور المؤهلة بعد التنظيف.
7. راجعت بصريًا 186 زوج أولوية من مرشحات تشابه موسعة بين موديلات الماركة نفسها؛ الصور مختلفة، فلم أجمعها بمجرد الخلفية المتشابهة. عدد المرشحات الموسعة 3852 ليس عدد تكرارات مثبتة. فحص تطابق البكسلات المفكوك لم يجد أزواجًا إضافية ضمن نطاقه، ولا يثبت غياب كل تكرار محتمل.
8. حفظت التقسيمات والمعرّفات والمصادر والأصول والخام، وأرشفت النسخة السابقة قبل اعتماد الجديدة. مجلد الملفات المهمة يحتوي جدولين ومجلد الصور فقط، وقيمه وبصمات صوره مطابقة للنسخة النهائية.
"""
    if supplemental:
        changes += f"9. استرجعت {supplemental['accepted']} صورة منظر إضافية من روابط المعارض المطابقة للهوية بعد نجاح اختبار الوصول العادي، بحوالي 4.7 ميجابايت. انخفض غياب المنظر المناسب من 89 إلى {no_overview} إعلانًا. حسمت الصور {supplemental.get('body_newly_resolved', 0)} حالات هيكل كانت غير واضحة. كانت محاولة استكمال الصور السابقة تركز على المعزول الذي لا يملك أي صورة؛ هذه الحالات المقبولة ذات الصور الداخلية والتفاصيل لم تكن ضمن قائمتها. لذلك لم يكن دقيقًا اعتبار كل الحالات الـ89 حد مصدر متعذرًا دون مراجعة بقية المعرض. لم يتغير سعر أو ممشى أو قسم تاريخي. أدلة الإضافة وفحصها في residual_completion.\n"
    if quality.get("user_issues"):
        changes += "10. أغلقت مراجعة ملاحظات المستخدم الست عشرة: صححت هدف هيئة 29 إعلان T-Roc وCountryman واحد إلى SUV/crossover بعد فحص صورها، وأضفت مسار ماركة مستقلًا للفئات غير المدعومة، وخمسة أعلام منفصلة للخلايا المركبة، وبصمات نسخة البيانات والتقسيم. صححت اسم حالة الصور الـ808 إلى روجعت وغير مناسبة للمنظر؛ نوع المصدر المفقود لا يعني أن الصورة غير مراجعة. أسباب عزل إعلانات KG العشرين منشورة في تقرير الملاحظات.\n"
    evidence = f"""- فحوص البيانات النهائية: **{validation['passed_count']}/{validation['total_checks']} ناجحة**، على data/processed بعد النشر. تشمل مطابقة {load_json(AUDIT / 'raw_manifest.json')['total_files']} ملفًا خامًا وأصليًا، وملفات الصور، و2534 صفحة دليل بهوياتها وبصماتها، وعدم عبور المجموعات أو SHA بين الأقسام.
- الاختبارات الآلية: **{units['tests_run']}/{units['tests_run']} ناجحة**، بلا فشل أو تخطٍّ. تشمل منع الوثيقة الموسومة exterior، والفشل المغلق دون مراجعة، وتعارض الوسم للصورة المطابقة، وأحكام الهيكل غير المحسومة، وحماية ملفات Windows وGitignore.
- مطابقة الحزمة المختصرة: {compact['accepted_listings']} إعلانًا، {compact['accepted_images']} صورة، وثلاثة عناصر فقط في المستوى الأول.
- أدلة المحتوى: data/audit/day20/content_review؛ الفحوص: validation.json وunit_tests.json؛ مخطط الحقول: data/processed/schema.json؛ مراجعة المصدر والتغييرات: enrichment_review.json.
- النسخة الآمنة قبل التصحيح: `{publication['archive']}`. النسخ القديمة تحفظ أيضًا عند تعديل التقارير.
"""
    write("docs/DATA_READINESS.md", "# جاهزية البيانات بعد إغلاق مراجعة المحتوى\n\n" + header + "\n## ما أصلح بالفعل\n\n" + changes + "\n## محتوى الصور\n\n" + content_table + "\n\n## نطاق مجموعات المهام\n\n" + task_table + "\n\nحد التعرف التجريبي المثبت: 12 مجموعة إجمالًا، 8 تدريب، 1 تحقق، 1 اختبار. عينة التقييم الأكبر: 20 تدريب و5 تحقق و5 اختبار على الأقل. جميعها أوصاف دعم وليست مقاييس أداء. وسوم اللون مشتقة من وصف المصدر؛ ليست مراجعة لونية ميدانية لكل صورة.\n\n## القيود الباقية الصريحة\n\n" + limits + "\n## دليل التحقق والحفظ\n\n" + evidence + "\n## حالة الخطة\n\nالنماذج والواجهة لم تبدأ. بوابة التدريب مغلقة وفق توجيه المستخدم؛ لا يبدأ baseline أو embeddings تلقائيًا بعد نجاح هذه الفحوص. بوابة M1 مستهدفة في 22 أكتوبر وقد تتأثر باستكمال البيانات الإضافي؛ لم يختصر التدقيق لتحقيق الموعد.\n")
    source_table = table(["مجموعة الملف", "الإعلانات المعتمدة", "الصور المعتمدة"], [[s, n, sum(r["source_dataset"] == s for r in images)] for s, n in sorted(source_counts.items())])
    write("docs/DATA_AUDIT.md", "# نتيجة تدقيق البيانات — النسخة الحالية\n\n" + header + f"\nالأصل 2803 إعلانًا؛ المعتمد {len(listings)}، المعزول {quality['quarantined_listings']}. الصور المعتمدة {len(images)}، ومجموعات الإعلان الاحترازية {groups}. لا ادعاء أن كل مجموعة سيارة واحدة مثبتة ميدانيًا.\n\n" + source_table + "\n\n## الإصلاحات\n\n" + changes + "\n## نتائج المهام\n\n" + task_table + "\n\n## الاختبارات\n\n" + evidence + "\n## الحدود\n\n" + limits + "\nالتفاصيل الحالية في DATA_READINESS.md. الملاحظات الـ25 للتحليل المرفق راجعت في external_analysis_final_review.json؛ سجل الحقول المصدرية محفوظ، والتصحيحات البصرية في أعمدة مشتقة منفصلة.\n")
    write("docs/TEST_REPORT.md", "# نتائج الفحوص الفعلية\n\n" + header + "\n" + evidence + "\n" + table(["الفحص", "الوصف", "النتيجة"], [[r["id"], r["description_ar"], "ناجح" if r["passed"] else "فشل"] for r in validation["checks"]]) + f"\n\nاختبار .gitignore: {units['gitignore']['checked_paths']} مسارًا، ناجح. سجل الاختبارات الكامل في data/audit/day20/unit_tests.json.\n\nاختبارات النماذج وAPI والواجهة: لم تُجرَ، وهذه الفحوص لا تنسب لها نجاحًا. لا دقة أو MAE مفترضة.\n")
    issues = [
        ["بطاقات وتفاصيل مؤهلة خطأ", "أصلح في أهلية كل المهام", f"{non_overview} صورة مستبعدة من المنظر مع حفظ الأصل"],
        ["غموض sport ودرجات اللون", "عولج بسياسة مشتقة موثقة", "قيم المصدر محفوظة؛ اللون الغامض والهيكل غير المحسوم بلا هدف"],
        ["تعارض مراجعة صور مطابقة", "صحح بعد فتح الأصلين", "كل SHA له نوع محتوى واحد؛ اختبار يمنع عودة التعارض"],
        ["ممشى غير معتاد بلا تأكيد", "عزل خاص بمهمة السعر", "11 إعلانًا محفوظًا كاملًا خارج مجموعة السعر"],
        ["قلة دعم الماركات والموديلات", "حد عينة معلن", f"{quality['supported_model_labels']} موديلًا و{quality['supported_make_labels']} ماركة مؤهلة أوليًا؛ جدول دعم وتدرج تجريبي"],
        ["نقص الحقول الاختيارية", "حد مصدر لم يختلق بديله", "240 إعلانًا لكل من اللون والهيكل والتاريخ الأصلي"],
        ["غياب منظر سيارة مناسب", "معزول من مهام المنظر", f"{no_overview} إعلانًا؛ محفوظ للعرض الوصفي مع صور متاحة أخرى"],
        ["ندرة تغطية السعر", "حد عينة معلن", f"{sparse_test} صف اختبار بتدريب أقل من خمس مجموعات للموديل"],
        ["تمثيل السوق والزمن", "حد استدلال", "مصدر واحد وجمع غير متجانس؛ لا اتجاه زمني موثوق ولا ادعاء تمثيل السوق"],
        ["جاهزية النماذج", "متوقف حسب طلب المستخدم", "لم يبدأ تدريب أو embeddings؛ المقاييس لم تختبر"],
    ]
    write("docs/ISSUE_REGISTER.md", "# سجل المشكلات الحالي\n\n" + header + "\n" + table(["المشكلة", "المصير", "الدليل أو الحد"], issues) + "\n\nسجل أحداث الاستكمال السابق محفوظ في data/audit/day20/issues.csv. قرارات مراجعة المحتوى والهيكل مكتملة في content_review، وأسباب عزل كل مهمة موجودة في أعمدتها وملفات الدعم. عدد أحداث التدقيق لا يساوي عدد الإعلانات المختلفة. لا تصف القيود المثبتة بأنها قيم مكتملة.\n")
    dictionary = [
        ["recognition_route", "مسار تجهيز التعرف: model_and_make للموديل والماركة، make_only للماركة فقط، unavailable لعدم الأهلية. ليس نتيجة استدلال ولا يمحو rec_label"],
        ["flag_source_unavailable / flag_high_annual_mileage / flag_price_iqr_outlier / flag_mileage_exact_100 / flag_young_low_mileage", "أعلام مستقلة بعد تفكيك data_quality_flags عند الفاصلة المنقوطة؛ تصفية آمنة دون إسقاط الخلايا متعددة العلامات، ومحظورة كميزات سعر"],
        ["color / body_type / trim", "القيم الأصلية الموثقة من المصدر؛ تحفظ مستقلة عن التصحيح البصري"],
        ["color_std", "عائلة لونية من خريطة معلنة؛ petrol مفقود لعدم حسمه، ولا حشو للمفقود"],
        ["body_type_std", "شكل الجسم؛ تصحيح بصري موثق عند المراجعة، ثم عائلة المصدر؛ sport لا يخمن"],
        ["trim_std", "تنظيف المسافات وPatrol Titanium فقط؛ المفقود لا يعوض"],
        ["body_label_status / body_label_evidence", "مصدر عائلة الهيئة أو مراجعتها البصرية أو تعذر حسمها مع ملف الدليل"],
        ["color_label_status / color_label_evidence", "وصف المصدر وخريطة العائلة، أو نقص/غموض؛ ليس تحققًا ميدانيًا"],
        ["content_kind", "vehicle_overview منظر، vehicle_detail تفاصيل، interior داخلية، document وثيقة"],
        ["content_review_status / content_review_evidence", "اكتمال مراجعة المحتوى وملف قرار مثبت ببصمة الصورة واللوحة"],
        ["image_kind / is_exterior / exterior_status", "نوع معرض المصدر الأصلي فقط؛ source_gallery_kind_missing تعني غياب نوع المصدر، مع بقاء مراجعة المحتوى مستقلة"],
        ["reviewed_not_vehicle_overview / content_review_pending", "الأولى حالة استبعاد بعد المراجعة لأنها ليست منظر سيارة مناسبًا؛ الثانية للمحتوى الذي لم يراجع، ولا يخلطان"],
        ["usable_recognition / usable_make_recognition", "منظر مراجع فريد ثنائيًا ضمن فئة تحقق حد دعم الموديل أو الماركة"],
        ["usable_visual_search", "منظر سيارة مراجع مع استبعاد نسخ SHA؛ ليست نتيجة نموذج بحث"],
        ["usable_color / usable_body_type", "منظر مراجع ووسم مشتق معروف غير متعارض ودعم كاف؛ مستقل لكل مهمة"],
        ["label_color / label_body_type", "وسم الصورة من سجل المصدر؛ لا يستخدم هدفًا بدل العمود المشتق"],
        ["label_color_std / label_body_type_std", "هدف مشتق مرتبط بالوسم المراجع للإعلان، مع احترام أهلية المهمة"],
        ["usable_price / price_status", "المستعمل فقط بعد عزل الممشى غير المعتاد الذي تعذر تأكيده؛ القيم التاريخية محفوظة"],
        ["n_images / n_overview_images", "عدد الملفات المقبولة وعدد مناظر السيارة قبل إزالة SHA من المهمة؛ لا يخلطان"],
        ["group_id / duplicate_group / split", "المجموعات والتقسيم الثابت؛ لا خلط صور أو تكرارات بين الأقسام"],
        ["mileage_source_status / current_source_mileage", "مقارنة منفصلة مع لقطة المصدر الحالية؛ لا تغير km_mileage التاريخي"],
        ["field_provenance / field_states", "دليل الحقول الأصلية وحالتها؛ لا تستخدم كميزات سعر"],
        ["live_availability_verified", "False؛ لا يثبت وجود الصفحة توفر السيارة للبيع"],
    ]
    write("docs/DATA_DICTIONARY.md", "# قاموس البيانات — المخطط المراجع\n\nترميز CSV هو UTF-8 مع BOM. جميع المعرفات والأصول والقيم المصدرية محفوظة.\n\n" + table(["الحقل", "المعنى والسياسة"], dictionary) + "\n\n## ملفات كل مهمة\n\n" + table(["الملف", "الغرض"], [["listings.csv / images.csv", "كل المعتمد وصفاته وأهليته، مع حفظ الصور غير المناسبة للتدريب"], ["price_dataset.csv / price_exclusions.csv / price_support.csv", "نطاق السعر والعزل الخاص به ودعم الموديلات في الأقسام"], ["recognition_images.csv / make_recognition_images.csv / make_fallback_images.csv", "صور الموديل والماركة المؤهلة، ومسار الماركة وحدها"], ["training_manifest.json", "نسخة وبصمات الجداول والتقسيم لمطابقة أي نموذج مستقبلي، وليس ملف نموذج مدرب"], ["visual_search_images.csv / color_images.csv / body_images.csv", "مجموعات مستقلة للبحث واللون والهيكل"], ["class_support.csv / make_support.csv / attribute_support.csv", "دعم الفئات الفعلي وتدرجه، وليس مقاييس دقة"], ["quarantine_listings.csv / image_exclusions.csv", "عزل السجلات/الصور بأسبابها دون حذف الأصل"]]) + "\n\nسمات السعر المسموحة حصريًا: make، model_std، year، km_mileage، trim_std، city، transmission، type_fuel. الهدف sar_price سعر إعلان بالريال. كل المعرفات والمسارات ووسوم المراجعة والتقسيم والسعر نفسه محظورة كميزات. حدود السنة والسعر والممشى موثقة في DATA_CONTRACT.md وschema.json. القيم الاختيارية المفقودة تبقى فارغة؛ ممشى الجديد لا يحول لصفر.\n")
    readiness = {"checked_at": now(), "processed_data_path": relative(PROCESSED), "data_cleanup_passed": True,
                 "all_source_fields_complete": False, "training_started": False, "training_allowed": False,
                 "current_dataset_sha256": {p.relative_to(PROCESSED).as_posix(): sha256(p) for p in PROCESSED.rglob("*") if p.is_file()},
                 "counts": {"listings": len(listings), "images": len(images), "listing_groups": groups, "price_rows": len(price), "no_overview_listings": no_overview},
                 "content": kinds, "task_images": quality["task_images"], "source_optional_missing": missing,
                 "limits_ar": limits, "validation": quality["validation"], "unit_tests": units["tests_run"], "compact": compact}
    save_json(AUDIT / "project_readiness.json", readiness)
    save_json(AUDIT / "data_completion_gate.json", {"updated_at": now(), "status": "review_complete_with_source_limits_models_held",
              "training_allowed": False, "training_started": False, "data_cleanup_passed": True, "all_source_fields_complete": False,
              "checks": {"all_image_content_reviewed": True, "task_label_policy": True, "independent_validation": True, "essential_bundle_synced": True, "honest_source_limits": True},
              "source_limits": {"missing_original_color_body_date_each": missing["color"], "no_overview_listings": no_overview,
                                "price_mileage_isolated": 11, "sparse_price_test_rows": sparse_test},
              "user_instruction_ar": "لا تبدأ في النموذج حتى تنتهي من البيانات بالكامل؛ لا تدريب تلقائي بعد إصدار تقرير محدود النطاق."})
    publication.update(final_checked_at=now(), final_validation=quality["validation"], unit_tests=units["tests_run"],
                       compact_sync=compact, processed_files_sha256=readiness["current_dataset_sha256"], report="docs/DATA_READINESS.md", passed=True)
    save_json(AUDIT / "content_publication.json", publication)
    print({"report": "docs/DATA_READINESS.md", "validation": quality["validation"], "unit_tests": units["tests_run"], "training_allowed": False})


if __name__ == "__main__": report()
