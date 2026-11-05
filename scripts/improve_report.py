"""تقرير تحسين عربي مرحلي/نهائي من الملفات المقاسة فقط."""
from improve_common import *


def fmt(x):return 'لم يختبر بعد' if x is None else f'{x:.4f}'


def report():
    s=session();checks=guard();experiments=read_csv(s/'experiments.csv') if (s/'experiments.csv').exists() else []
    table=['| المهمة | val قبل: دقة / macro-F1 | val بعد: دقة / macro-F1 | test قبل: دقة / macro-F1 | test بعد: دقة / macro-F1 |',
           '|---|---|---|---|---|'];details=[];completed=[]
    for task,name in [('make','الماركة'),('model','الموديل'),('color','اللون'),('body','الهيكل')]:
        folder=s/task;val=load_json(folder/'val_metrics.json',{});test=load_json(folder/'test_metrics.json',{})
        before=val.get('before') or load_json(folder/'val_before_metrics.json',{})
        cell=lambda m:'لم يختبر بعد' if not m else f'{m["accuracy"]*100:.2f}% / {m["macro_F1"]:.4f}'
        table.append(f'| {name} | {cell(before)} | {cell(val)} | {cell(test.get("before",{}))} | {cell(test)} |')
        selected=load_json(folder/'selected.json',{})
        text=f'### {name}\n\nاختيار val: `{json.dumps(selected,ensure_ascii=False)}`.\n\n'
        for split,m in [('val',val),('test',test)]:
            if m:
                ci=m['ci95'];text+=f'{split}: {m["n"]} صورة. فاصل 95% للدقة [{ci["accuracy"][0]*100:.2f}%, {ci["accuracy"][1]*100:.2f}%]؛ macro-F1 [{ci["macro_F1"][0]:.4f}, {ci["macro_F1"][1]:.4f}]. إعادة أخذ عينات {ci["groups"]} مجموعة group_id، 2000 تكرار، مع فئات ثابتة؛ ليست صورًا مستقلة.\n\n'
                if m.get('paired_delta_ci95'):
                    ci=m['paired_delta_ci95']['macro_F1'];text+=f'فاصل فرق macro-F1 المقترن بعد−قبل: [{ci[0]:.4f}, {ci[1]:.4f}]. احتواء الصفر يعني أن التحسن العددي وحده لا يحسم فرقًا ثابتًا.\n\n'
                text+=f'خط الأكثر تكرارًا من التدريب: `{m["train_majority_baseline"]["class"]}`، دقته {m["train_majority_baseline"]["accuracy"]*100:.2f}%.\n\n' if m.get('train_majority_baseline') else ''
        for split in ['val','test']:
            worst=read_csv(folder/f'{split}_worst10.csv') if (folder/f'{split}_worst10.csv').exists() else []
            if worst:
                text+=f'أضعف الفئات على {split} (حد أقصى 10؛ الهيكل له 5 فئات فقط):\n\n| الفئة | precision | recall | F1 | الدعم | أكثر اختلاط | سبب محتمل |\n|---|---|---|---|---|---|---|\n'
                for r in worst:text+=f'| {r["class"].replace("|"," / ")} | {float(r["precision"]):.3f} | {float(r["recall"]):.3f} | {float(r["F1"]):.3f} | {r["eval_support"]} | {r["most_confused_with"].replace("|"," / ")} | {r["possible_reason_ar"]} |\n'
                text+='\n'
        details.append(text)
        if test:completed.append(task)
    price_val=load_json(s/'price/val_metrics.json',{});price_test=load_json(s/'price/test_metrics.json',{})
    price_before=load_json(s/'price/val_before_metrics.json',{})
    price_text='## السعر\n\nلا precision أو recall أو F1 للانحدار؛ نعرض MAE وMAPE وRMSE وR² وشرائح الموديل بدل مقاييس تصنيف لا تنطبق.\n\n| القسم | قبل MAE / MAPE | بعد MAE / MAPE | CI95 للـMAE بعد |\n|---|---|---|---|\n'
    price_notes=[]
    for split,m,before in [('val',price_val,price_before),('test',price_test,price_test.get('before',{}))]:
        if m:
            ci=m['ci95']['MAE_SAR'];price_text+=f'| {split} | {before["MAE_SAR"]:.2f} ريال / {before["MAPE_percent"]:.2f}% | {m["MAE_SAR"]:.2f} ريال / {m["MAPE_percent"]:.2f}% | [{ci[0]:.2f}, {ci[1]:.2f}] |\n'
            ci=m['ci95']['MAPE_percent'];price_notes.append(f'فاصل MAPE على {split}: [{ci[0]:.2f}%, {ci[1]:.2f}%]؛ n={m["n"]}.')
    price_text+='\n'+'\n\n'.join(price_notes)+'\n\n'
    val_median=load_json(s/'price/val_median_baseline.json',{})
    if val_median:
        price_text+=f'خط وسيط كل (ماركة، موديل) على val: MAE={val_median["MAE_SAR"]:.2f} ريال، MAPE={val_median["MAPE_percent"]:.2f}%؛ فاصل MAE 95%={val_median["ci95"]["MAE_SAR"]}. تعلم من train فقط، مع رجوع للشركة ثم الوسيط العام.\n\n'
    if price_test:
        completed.append('price');b=price_test['median_baseline'];ci=price_test['paired_delta_MAE_ci95']
        price_text+=f'\nوسيط كل (ماركة، موديل) متعلم من التدريب فقط، مع رجوع للشركة ثم الوسيط العام إن غاب الموديل: MAE={b["MAE_SAR"]:.2f}، MAPE={b["MAPE_percent"]:.2f}%. هذا تعريف مستقل عن الوسيط القديم للموديل والسنة (16571.16) وعن الرقم 16746 المقدم قبل التحقق. فاصل فرق MAE المقترن [{ci[0]:.2f}, {ci[1]:.2f}].\n\n'
    if (s/'price/group_cv_results.json').exists():price_text+='GroupKFold بخمس طيات داخل train فقط، وفحص عدم تقاطع المجموعات لكل طية؛ مفردات الفئات تتعلم مجددًا من جزء التدريب لكل طية. الاختيار بالـval MAE فقط؛ CV تشخيص استقرار ولا يدخل test. الهدف log(price)، والسمات الثماني من العقد؛ لا flag_* أو status أو معرفات. لم تملأ قيم ممشى أو لون أو سنة ولم يدخل الجديد نموذج السعر.\n\n'
    if (s/'price/test_worst10.csv').exists():
        price_text+='أسوأ عشرة موديلات حسب MAE (قلة العينات معلنة؛ ليست مقارنة إحصائية بين الموديلات):\n\n| الموديل | n | MAE | MAPE | سبب محتمل |\n|---|---|---|---|---|\n'
        for r in read_csv(s/'price/test_worst10.csv'):price_text+=f'| {r["class"].replace("|"," / ")} | {r["n"]} | {float(r["MAE_SAR"]):.2f} | {float(r["MAPE_percent"]):.2f}% | {r["possible_reason_ar"]} |\n'
    search_text='## البحث البصري\n\nالمعرض train فقط، مع استبعاد listing_uid وsha256 وgroup_id صراحة ومنع تكرار المجموعة في النتائج. الملاءمة حسب rec_label، وليست إثباتًا لهوية السيارة أو حكمًا بشريًا لكل زوج.\n\n'
    for split in ['val','test']:
        m=load_json(s/f'search/{split}_metrics.json',{})
        if m:
            old_search=load_json(s/f'search/{split}_before_metrics.json',{})
            if old_search:search_text+=f'قبل التحسين على {split}: top-1={old_search["top1"]*100:.2f}%، top-5={old_search["top5"]*100:.2f}%. CI95 لـtop-1={old_search["ci95"]["top1"]} ولـtop-5={old_search["ci95"]["top5"]}.\n\n'
            search_text+=f'{split}: عدد الاستعلامات {m["n_query_groups"]}؛ top-1={m["top1"]*100:.2f}% وtop-5={m["top5"]*100:.2f}%، وtop-1 macro-F1={m["top1_macro_F1"]:.4f}. CI95 لـtop-1={m["ci95"]["top1"]} ولـtop-5={m["ci95"]["top5"]} بوحدة المجموعة. top-5 المرجعي {m["reference_top5"]*100:.2f}%. لا ننسب ثبات الرقم لتحسن جديد.\n\n'
            search_text+=f'فاصل macro-F1 للبحث 95%={m["macro_F1_ci95"]}؛ تفاصيل precision وrecall وF1 وفواصلها لكل فئة محفوظة في search/{split}_per_class.csv.\n\n'
            if split=='test':completed.append('search')
    failed=[r for r in experiments if r['accepted'].lower()=='false']
    failures='## التجارب التي لم تعتمد\n\nالرفض بسبب عدم تحسن معيار val؛ لا نخفي المحاولات ولا نتراجع عن الاختيار بسبب test. السجل الكامل يشمل كل قبول ورفض وبصمات الملفين ووقت التجربة.\n\n| المهمة | التجربة | قبل | بعد | التغيير |\n|---|---|---|---|---|\n'
    for r in failed:failures+=f'| {r["task"]} | {r["experiment"]} | {float(r["before"]):.5f} | {float(r["after"]):.5f} | {r["single_change"].replace("|"," / ")} |\n'
    protocol=load_json(s/'protocol.json');locks={task:load_json(s/task/'test_once.json',{}) for task in ['make','model','color','body','price','search']}
    reviewed=load_json(s/'val_error_review_samples.json',[])
    visual_text='## مراجعة فعلية لعينات أخطاء val\n\n'
    for r in reviewed:
        visual_text+=f'- [{r["image_uid"]}]({Path(r["path"]).as_posix()}): وسم {r["actual"]} مقابل تنبؤ {r["predicted"]}. {r.get("visual_observation_ar", "لم يكتمل فحصها بصريًا")}\n'
    if reviewed:
        visual_text+='\nالهوية في الجدول Mazda CX-90، وتعريف [Mazda الرسمي](https://news.mazdausa.com/2023-02-07-Mazda-Announces-Pricing-and-Packaging-For-First-Ever-2024-Mazda-CX-90) يصف العائلة بأنها SUV، بينما وسمها المجمد van. هذه قرينة قوية على ضوضاء وسم المصدر في العينة، لا مبرر لتعديل الوسم سرًا أو حذف الخطأ. المقاييس تقيس المطابقة للوسوم المجمدة ولا تساوي تحققًا ميدانيًا من كل لون أو هيئة. لم تستعمل هذه المراجعة صور test.\n\n'
    final=all(locks[t].get('state')=='completed' for t in locks)
    units=load_json(ROOT/'data/audit/day20/unit_tests.json',{})
    qa=load_json(s/'release/qa.json',{})
    checklist=[{'check_ar':'مفاتيح وأقسام ومجموعات وأعلام النسختين متطابقة','status':'ناجح' if not checks['differences'] else 'راسب','detail':str(len(checks['differences']))},
        {'check_ar':'بصمة التقسيم ثابتة','status':'ناجح' if checks['split_matches_manifest'] else 'راسب','detail':checks['split_fingerprint']},
        {'check_ar':'النماذج السابقة محفوظة ببصماتها','status':'ناجح','detail':'فحص guard يوقف عند تغير نموذج مرجعي'},
        {'check_ar':'دقة كل الوسوم الدلالية مثبتة','status':'راسب: لا تحقق شامل','detail':'عينة val لـCX-90 موسومة van مقابل تعريف SUV؛ وثقت ولم تعدل'},
        {'check_ar':'اختيار val مقفل قبل الاختبار','status':'ناجح' if (s/'selection_lock.json').exists() else 'قيد التنفيذ','detail':'selection_lock.json'},
        {'check_ar':'حدث test نهائي واحد لكل مهمة','status':'ناجح' if final else 'قيد التنفيذ','detail':json.dumps({k:v.get('state','not_started') for k,v in locks.items()})},
        {'check_ar':'اختبارات الكود الآلية','status':'ناجح' if units.get('passed') else 'راسب','detail':str(units.get('tests_run','لم يختبر'))},
        {'check_ar':'استدلال الحزمة في عمليات جديدة','status':'ناجح' if qa.get('passed') else 'قيد التنفيذ','detail':'صور val فقط؛ لا إعادة لتقييم test'},
        {'check_ar':'تقييم خارجي فعلي بـ20–30 صورة','status':'راسب: لم يختبر بعد','detail':'external_test/images فارغ؛ السكربت والتعليمات جاهزان ولا نتائج مخترعة'}]
    save_csv(s/'checklist.csv',checklist)
    body=f'''<div dir="rtl" lang="ar">

# تحسين نماذج سعّر — 21 أكتوبر 2026

**الحالة:** {'اكتملت التجارب والتقييم النهائي المسجل' if final else 'العمل جارٍ؛ النتائج المتاحة من val فقط، ولا وعود بنتائج test'}. آخر إصدار: {now()} UTC.

## سلامة البيانات والتقييم

اعتمد المستخدم الملفين المختصرين بعد تأكيد بصمتيهما خارجيًا: listings SHA=`{checks['files']['listings.csv']['sha256']}`، images SHA=`{checks['files']['images.csv']['sha256']}`. manifest الأصلي للجداول الكاملة لم يعدل. قارنت المفاتيح وsplit وgroup_id وusable_* برمجيًا: {len(checks['differences'])} اختلافات. لا تعديل قيم أو أعلام أو تقسيم؛ بصمة التقسيم `{checks['split_fingerprint']}`.

الأرقام المرجعية للصورة الواحدة محفوظة قبل التطوير: الماركة 53.91% والموديل 71.52% واللون 70.09% والهيكل 90.56%، والسعر MAE=13088.80 والبحث top-5=67.20%. الملفات القديمة والنماذج الحالية لم تستبدل. سجل التجارب يسجل SHA للملفين في كل صف.

النطاق ثابت: 59 فئة موديل فقط، و26 ماركة و10 ألوان و5 أنواع هيكل. لا تنطبق دقة الموديل على الموديلات الـ272 كلها. مسار الماركة يغطي المؤهل لها، وعتبة القرار لا تثبت كشف كل موديل أو ماركة أو صورة خارج الفئات المدربة.

المصدر منصة إعلانات واحدة ولقطة جمع أكتوبر 2026؛ لا عينة سوق شاملة ولا سلسلة أسعار عبر الزمن. بقيت الجداول 2742 إعلانًا و13416 صورة، ومجموعة السعر 2293 مستعملة؛ كل مهمة بصرية تستخدم أهلية usable_* الخاصة بها لا الصور كلها.

**قيد منهجي:** نتائج test السابقة كانت معروفة قبل هذه الجولة. اخترت التغييرات على val فقط وقفلت النماذج قبل حدث اختبار نهائي واحد لكل مهمة؛ هذا لا يحول test إلى مجموعة عمياء جديدة، ولا يعد اختبار تعميم مستقلًا. لا إعادة اختيار بعد رؤية test، ولو انخفض الرقم.

فواصل bootstrap مشروطة بالنموذج المختار ولا تصحح تفاؤل اختيار المعاملات على val أو تعدد التجارب. تمثل عدم اليقين داخل مجموعات هذه العينة ومصدرها، لا التعميم على السوق أو الهاتف. أعيد استخدام عينات المجموعات نفسها للمقارنة المقترنة قبل/بعد.

التحسين تدريجي وبذرة 42 وCPU، مع تشغيل طبقات أكثر ثم وزن فئات وlabel smoothing وتعزيز أفقي معتدل ومعدل تعلم أقل وTTA كخيارات منفصلة. إحصاءات BatchNorm مجمدة؛ التمثيلات المسبقة لتدريب/val فقط وتخزن float32. التجهيز القياسي للشبكة يعيد التحجيم لمدخلها لكنه لا يغير الصور الأصلية أو يختلق تفاصيل عالية الدقة.

## قبل وبعد — الصورة الواحدة

{chr(10).join(table)}

الماركة المشتقة من الموديل مع الرجوع المستقل اختيرت بعتبة val دون استخدام الموديل الحقيقي للصورة الجديدة. المقارنة التي توجه حسب usable_recognition من الجدول محفوظة باسم metadata_route/oracle للتحليل وحده؛ تعتمد معلومات أهلية معروفة ولا تسوق كآلية تعرف صورة مجهولة. درجات hard المشتقة 0/1 تمثل قرارًا فئويًا وليست احتمال صحة 100%.

{chr(10).join(details)}

{price_text}

{search_text}

{failures}

{visual_text}

## تحقق التشغيل

الاختبارات الآلية الحالية: {units.get('tests_run','لم يختبر')}، ناجحة: {units.get('passed',False)}. فحص الحزمة من عمليات جديدة: {qa.get('passed','لم يختبر بعد')}؛ قياس التشغيل يستخدم صورة val فقط ولا يكرر test. البصمات والمقاييس وأحداث الاختبار محفوظة، ولم تستبدل مؤشرات النماذج الحالية.

## الاختبار الخارجي والداشبورد

جهزت external_test/images وlabels.csv وتعليمات صور فعلية 20–30 من الجوال أو مصدر آخر مع وسوم يدوية. شغلت سكربت التقييم على المجلد الفارغ فأعلن «لم يختبر بعد» ولم ينشئ صورًا أو دقة. أمره `.venv/Scripts/python -X utf8 scripts/improve_external.py evaluate`.

في الداشبورد: نوع الهيكل (سيدان/SUV...) ليس كشف ضرر الهيكل. لا توجد وسوم موضع ضرر أو شدته أو أثره السعري، فلا أدعي تدريب بطاقات السلامة أو خصم 15% أو نسبة ضرر. كما أن bootstrap هنا لفاصل ثقة أداء مثل MAE، وليس فاصل تنبؤ سعر لسيارة فردية؛ لا يملأ بطاقة نطاق السعر تلقائيًا. ولا تدعم لقطة جمع واحدة منحنى تطور سوق عبر سنوات. صور الإعلان لا تثبت التوفر الحي أو سعر البيع النهائي.

## الملفات وقائمة التحقق

مجلد العمل والنماذج المختارة: [جلسة التحسين](C:/CAR/{s.relative_to(ROOT).as_posix()})؛ [التجارب كاملة](C:/CAR/{s.relative_to(ROOT).as_posix()}/experiments.csv)؛ [قائمة التحقق](C:/CAR/{s.relative_to(ROOT).as_posix()}/checklist.csv)؛ [تعليمات الاختبار الخارجي](C:/CAR/external_test/README.md).

بيانات precision وrecall وF1 لكل الفئات، ومصفوفات الخلط وملفات أسوأ الفئات وفواصل الثقة متاحة لكل مهمة وقسم داخل الجلسة. لا أختلق عشر فئات للهيكل ذي الخمس فئات. ملفات السعر تعرض شرائح وأخطاء بدل precision/recall غير المنطبقين على الانحدار.

المراجع التقنية: [الضبط ونقل التعلم في PyTorch](https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial.html)، [GroupKFold الرسمي](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupKFold.html).

</div>
'''
    save_bytes(ROOT/'docs/MODEL_IMPROVEMENT_2026-10-21.md',body.encode('utf-8'))
    print('REPORT',len(completed),'tasks final / 6;',len(experiments),'experiments',flush=True)


if __name__=='__main__':report()
