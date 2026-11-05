"""توثيق ومراجعة المرشح متعدد الصور دون تشغيل الاختبار التاريخي."""
import argparse
import importlib.util
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
import numpy as np
from image_price_multiview import ROOT, BASE, rows, js, csvwrite, digest, validate, met, bootstrap, gate


def reference_predictions(run, d, val):
    """مرجع مقفل وGemini على val فقط، من مخزن البكسلات المجمد."""
    f = {k: np.asarray(np.load(run / 'cache' / (k + '.npy'), mmap_mode='r')[val], dtype=np.float64)
         for k in ('crop', 'full', 'x53')}
    with np.load(BASE / 'gemini/v2_model_artifacts.npz') as a:
        v = (a['Vq'].astype(np.float32) * a['scale_v']).astype(np.float64)
        zg = a['b_tot'][0] + f['crop'] @ a['w_crop_7x4'] + f['full'] @ a['w_full_7x4'] + f['x53'] @ a['w_53_lin'] + ((f['x53'] @ v) ** 2) @ a['lk']
        zg = np.clip(zg, *a['log_prediction_bounds'])
    with np.load(BASE / 'head.npz') as h:
        sim = np.clip(f['x53'] @ h['support53'].astype(np.float64).T, -1, 1)
        z = h['intercept'][0] + f['crop'] @ h['w_crop'] + f['full'] @ h['w_full'] + sim ** 2 @ h['poly_coef'] + np.exp(-1.5 * np.maximum(2 - 2 * sim, 0)) @ h['rbf_coef']
    bounds = js(BASE / 'policy.json')['bounds']
    return np.exp(np.clip(.75 * np.clip(z, *bounds) + .25 * zg, *bounds)), np.exp(zg)


def weighted_metrics(y, p, weights):
    w = weights / weights.sum(); e = np.abs(y - p)
    return {'MAE_SAR': float(w @ e), 'MAPE_percent': float(100 * (w @ (e / y))),
            'within_20_percent': float(100 * (w @ (e / y <= .2))),
            'RMSE_SAR': float(np.sqrt(w @ (e ** 2))), 'n_images': len(y)}


def exact_cached_head(run, val):
    """تنبؤ الرأس المقفل من الميزات المخزنة بحساب double، دون Gram float32.

    مستقل عن فك الصور وforward اللذين يستخدمهما مشغل التسليم. لا يغيّر أي وزن.
    """
    raw = np.asarray(np.load(run / 'cache/raw.npy', mmap_mode='r')[val], dtype=np.float64)
    with np.load(run / 'head.npz') as h:
        v = np.clip((raw - h['vs_mean']) / h['vs_std'], -5, 5)
        v /= np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12)
        def adjusted(name):
            x = np.asarray(np.load(run / 'cache' / (name + '.npy'), mmap_mode='r')[val], dtype=np.float64)
            if name != 'full':
                x[:, :-154] *= np.sqrt(1 - .1 / 1.95) / np.maximum(np.linalg.norm(x[:, :-154], axis=1, keepdims=True), 1e-12)
                x[:, -154:] = v * np.sqrt(.1 / 1.95)
                x /= np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
            return x
        cr = adjusted('crop'); fu = adjusted('full'); x = adjusted('x53')
        sim = np.clip(x @ h['support53'].astype(np.float64).T, -1, 1)
        z = h['intercept'][0] + cr @ h['w_crop'] + fu @ h['w_full'] + (sim ** 2) @ h['poly_coef'] + np.exp(-1.5 * np.maximum(2 - 2 * sim, 0)) @ h['rbf_coef']
    return np.exp(np.clip(z, *js(run / 'policy.json')['bounds']))


def finish(run):
    validate(run)
    if (run / 'RESULTS.json').exists(): raise ValueError('النتائج النهائية موجودة؛ لا تعاود تقييم المرشحين')
    d = rows(run / 'images.csv'); cv = js(run / 'CV_LOCK.json')
    val = np.array([i for i, r in enumerate(d) if r['split'] == 'val'])
    cover = np.array([d[i]['is_cover'] == 'True' for i in val]); y = np.array([float(d[i]['price_sar']) for i in val])
    groups = np.array([d[i]['group_id'] for i in val]); original, gemini = reference_predictions(run, d, val)
    single_rows = rows(run / 'single_val_head_predictions.csv'); multi_rows = rows(run / 'multi_val_head_predictions.csv')
    single = np.array([float(r['predicted_sar']) for r in single_rows]); multi = np.array([float(r['predicted_sar']) for r in multi_rows])
    if [r['image_uid'] for r in single_rows] != [d[i]['image_uid'] for i in val]: raise ValueError('اختلاف ترتيب توقعات التحقق')
    bounds = js(run / 'policy.json')['bounds']
    single = np.exp(np.clip(.75 * np.log(single) + .25 * np.log(gemini), *bounds))
    approximate = np.exp(np.clip(.75 * np.log(multi) + .25 * np.log(gemini), *bounds))
    exact = exact_cached_head(run, val)
    expected = np.exp(np.clip(.75 * np.log(exact) + .25 * np.log(gemini), *bounds))
    js(run / 'kernel_rounding_diagnostic.json', {'maximum_float32_gram_vs_double_cached_sar': float(np.abs(expected - approximate).max()),
        'head_sha256': digest(run / 'head.npz'), 'weights_changed': False,
        'explanation': 'معاملات SVR كبيرة ومتعاكسة؛ تقريب حاصل الضرب إلى float32 قبل الجمع يراكم خطأ. مرجع التشغيل الصحيح يستخدم ميزات المخزن ووزن الرأس ذاته بحساب double؛ حد التطابق لم يتغير.'})
    spec = importlib.util.spec_from_file_location('mv_runtime', run / 'runtime.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module); rt = module.ImagePriceRuntime(run)
    actual = []; scores = []; tt = time.perf_counter()
    for start in range(0, len(val), 8):
        ix = val[start:start + 8]
        p, s = rt.predict_batch_paths([ROOT / 'الملفات_المهمة' / d[i]['relative_path'] for i in ix])
        actual.extend(p); scores.extend(s)
        if start % 320 == 0: print('VERIFY val', start + len(ix), '/', len(val), flush=True)
    actual = np.array(actual); scores = np.array(scores)
    diff = np.abs(actual - expected); relative = diff / np.maximum(expected, 1)
    verify = {'n': len(val), 'max_difference_sar': float(diff.max()), 'max_relative_difference': float(relative.max()),
              'tolerance_sar': 10., 'tolerance_relative': .0001, 'seconds': time.perf_counter() - tt,
              'passed': bool(diff.max() < 10 and relative.max() < .0001)}
    js(run / 'runtime_verification.json', verify)
    if not verify['passed']: raise ValueError('فشل إعادة إنتاج توقعات الرأس من مشغل الصور')
    try:
        rt.predict(ROOT / 'REQ.md')
        raise AssertionError('قبول ملف غير صورة')
    except ValueError: pass
    actual_rows = [{'image_uid': d[i]['image_uid'], 'listing_uid': d[i]['listing_uid'], 'group_id': d[i]['group_id'],
        'is_cover': d[i]['is_cover'], 'actual_sar': float(y[j]), 'original_sar': float(original[j]),
        'single_control_sar': float(single[j]), 'multi_candidate_sar': float(actual[j]),
        'domain_score': float(scores[j]), 'accepted': bool(scores[j] <= rt.gem.domain_threshold)} for j, i in enumerate(val)]
    csvwrite(run / 'val_predictions.csv', actual_rows)
    metrics = {name: {**met(y[cover], p[cover]), **bootstrap(y[cover], p[cover], original[cover], groups[cover])}
               for name, p in [('original', original), ('single_control', single), ('multi_candidate', actual)]}
    paired_control = bootstrap(y[cover], actual[cover], single[cover], groups[cover])
    train_covers = [r for r in d if r['split'] == 'train' and r['is_cover'] == 'True']
    boundaries = np.quantile([float(r['price_sar']) for r in train_covers], [1 / 3, 2 / 3])
    decisions = {'train_cv': cv['practical_gate'],
        'val_vs_frozen_deployed': gate(y[cover], actual[cover], original[cover], boundaries[1]),
        'val_vs_single_control': gate(y[cover], actual[cover], single[cover], boundaries[1])}
    passed = all(all(g.values()) for g in decisions.values())
    bands = []
    for k, label in enumerate(['الأرخص', 'المتوسطة', 'الأغلى']):
        ix = cover & (np.digitize(y, boundaries) == k)
        for name, p in [('original', original), ('single_control', single), ('multi_candidate', actual)]:
            bands.append({'band': label, 'model': name, **met(y[ix], p[ix])})
    csvwrite(run / 'price_bands.csv', bands)
    counts = __import__('collections').Counter(d[i]['listing_uid'] for i in val)
    w = np.array([1 / counts[d[i]['listing_uid']] for i in val])
    secondary = {name: weighted_metrics(y, p, w) for name, p in [('original', original), ('single_control', single), ('multi_candidate', actual)]}
    # Match archived 321 primary predictions by their original SHA/sample ID linkage.
    staged = rows(ROOT / 'models/claude_image_price_tuning_20261022/train_val/dataset.csv')
    bysample = {r['sample_id']: r['sha256'] for r in staged if r['split'] == 'val'}
    bysha = {d[i]['sha256']: original[j] for j, i in enumerate(val) if cover[j]}
    archived_delta = max(abs(float(r['predicted_sar']) - bysha[bysample[r['sample_id']]]) for r in rows(BASE / 'val_predictions.csv'))
    if archived_delta > 10: raise ValueError('فشل إعادة إنتاج المرجع القديم')
    accepted = scores[cover] <= rt.gem.domain_threshold
    result = {'primary_val': metrics, 'paired_multi_vs_control': paired_control,
        'primary_n': int(cover.sum()), 'secondary_all_views_equal_listing_weight': secondary,
        'internal_practical_gate': decisions, 'passes_all_internal_gates': passed,
        'acceptance_rate_percent': float(accepted.mean() * 100),
        'accepted_only_metrics': met(y[cover][accepted], actual[cover][accepted]) if accepted.any() else None,
        'domain_threshold_unchanged': rt.gem.domain_threshold, 'domain_not_price_confidence': True,
        'archived_reference_reproduction_max_sar': archived_delta,
        'test_evaluations_this_round': 0, 'external_status': 'لم يختبر بعد',
        'decision': 'مرشح تحسن داخلي؛ يحتاج تقييمًا خارجيًا قبل الاعتماد' if passed else 'لم يجتز شروط التحسن؛ الاحتفاظ بالنموذج الحالي وإيقاف إضافة التعقيد',
        'DINOv2_started': False, 'current_pointer_changed': False}
    js(run / 'RESULTS.json', result)
    prepare_external(run)
    shutil.copytree(ROOT / 'scripts', run / 'source_scripts', ignore=shutil.ignore_patterns('__pycache__'))
    js(run / 'source_sha256.json', {p.relative_to(run).as_posix(): digest(p) for p in (run / 'source_scripts').rglob('*') if p.is_file()})
    names = ['runtime.py', 'policy.json', 'head.npz', 'encoder_weights.pth', 'vendor/gemini_features.py',
             'gemini/runtime_policy.json', 'gemini/v2_model_artifacts.npz', 'gemini/domain.joblib', 'protocol.json', 'CV_LOCK.json', 'source_sha256.json']
    js(run / 'LOCK.json', {'sha256': {n: digest(run / n) for n in names}, 'test_not_evaluated_this_round': True})
    # Fresh working directory/process, exactly one fixed val image, after artifact lock.
    image = ROOT / 'الملفات_المهمة' / d[val[np.flatnonzero(cover)[0]]]['relative_path']
    process = subprocess.run([str(ROOT / '.venv/Scripts/python.exe'), '-X', 'utf8', str(run / 'runtime.py'),
                              '--bundle', str(run), '--image', str(image)], cwd=ROOT / 'tmp', capture_output=True, text=True, encoding='utf-8')
    js(run / 'fresh_process_check.json', {'exit_code': process.returncode, 'stdout': process.stdout, 'stderr': process.stderr})
    if process.returncode: raise ValueError('فشل التشغيل من عملية مستقلة')
    validate(run)
    checklist = [{'item': 'ثبات البيانات والتقسيم والأوزان القديمة', 'result': 'ناجح'},
        {'item': 'اختيار C داخل GroupKFold train فقط', 'result': 'ناجح'},
        {'item': 'وزن إجمالي متساو لكل إعلان', 'result': 'ناجح'},
        {'item': 'استدلال بصورة واحدة وتطابق التشغيل', 'result': 'ناجح'},
        {'item': 'عدم تقييم test التاريخي', 'result': 'ناجح'},
        {'item': 'اجتياز شروط التحسن الداخلي المحددة مسبقًا', 'result': 'ناجح' if passed else 'راسب'},
        {'item': 'إثبات التعميم الخارجي', 'result': 'لم يختبر بعد'},
        {'item': 'إثبات عدم وجود أي سيارة مكررة بزوايا مختلفة', 'result': 'غير مثبت؛ الفحص محدود'}]
    csvwrite(run / 'checklist.csv', checklist)
    report(run, result, cv, bands)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


def prepare_external(run):
    folder = ROOT / 'external_test/image_price_multiview'
    folder.mkdir(exist_ok=True)
    for part in ('calibration', 'evaluation'):
        (folder / part / 'images').mkdir(parents=True, exist_ok=True)
        p = folder / part / 'labels.csv'
        if not p.exists(): p.write_text('image_path,actual_sar,group_id,source_url,source_name,price_date\n', encoding='utf-8-sig')
    (folder / 'README_AR.md').write_text('''# اختبار خارجي مستقل لسعر الإعلان من صورة واحدة

لم يُجمع أو يُقيّم أي مثال خارجي في هذه الجولة. الصور والنتائج لا تُختلق.

اجمع أسعار طلب موثقة تخص نفس السيارة الظاهرة في الصورة، لا متوسط سعر السوق. سجّل المصدر وتاريخ السعر. استخدم مصادر وبائعين وزوايا مختلفة، وتحقق أن السيارة أو صورها ليست في بيانات المشروع. كل صور السيارة أو إعادة نشر إعلانها تحمل group_id واحدًا.

ابدأ بعينة استطلاعية من 20–30 إعلانًا لكشف مشاكل التشغيل فقط. للتقييم الأكثر فائدة استهدف 100–200 إعلان مستقل: جزء للمعايرة وجزء منفصل للتقييم، مثل 80 معايرة و120 تقييم إن توفرت. ليست هذه الأعداد ضمانًا للدقة؛ اعرض فواصل الثقة وحجم كل شريحة. لا تجعل هوية المصدر ملتبسة تمامًا مع دور العينة؛ مثّل أكثر من مصدر في كل مجموعة مع فصل السيارات والبائعين قدر الإمكان.

ضع صور المعايرة في calibration/images، والتقييم في evaluation/images، وأكمل ملف labels.csv لكل مجموعة. كل صف صورة واحدة محددة سلفًا لإعلان. السعر سعر إعلان، وليس سعر بيع فعلي. لا تملأ سنة أو ممشى أو حالة تخمينيًا.

تضبط العتبة على المعايرة فقط، بمقارنة الخطأ الفعلي مع درجة النطاق وعرض منحنى نسبة القبول مقابل MAE وMAPE ونسبة ضمن ±20%. إن لم تساعد الدرجة على انتقاء توقعات أفضل، لا تدّع نجاح الرفض ولا ترفع العتبة لمجرد قبول صور أكثر. ثبّت العتبة ثم شغّل مجموعة التقييم مرة واحدة. يجب عرض مقاييس جميع التوقعات تشخيصيًا، ونسبة القبول ومقاييس المقبول معًا. لا تستخدم نتيجة التقييم لإعادة ضبط العتبة.

درجة النطاق الحالية احترازية وغير معايرة؛ لا تثبت صحة السعر. فاصل bootstrap لمتوسط الخطأ ليس نطاق سعر سيارة فردية. لا يوجد نطاق سعر فردي معاير حاليًا.
''', encoding='utf-8')


def report(run, result, cv, bands):
    audit = js(run / 'data_audit.json'); out = ['# تجربة التدريب متعدد الصور — سعر الإعلان من صورة واحدة', '',
        result['decision'], '', '## ما جرى فعليًا', '',
        f"- صور التدريب: {audit['counts_images']['train']}، لإعلانات مستقلة: {audit['counts_listings']['train']}. صور val: {audit['counts_images']['val']}، لإعلانات: {audit['counts_listings']['val']}.",
        '- لم تتغير بيانات المصدر أو split أو group_id أو الأعلام. الاستبعاد محلي لهذه التجربة فقط، إن ظهر في التدقيق.',
        '- ResNet18 وميزات القص/الصورة الكاملة/الانعكاس بقيت مجمدة كما في النموذج السابق. الذي تدرب هو رأس SVR على لوغاريتم السعر، باستخدام صور أكثر.',
        '- مجموع أوزان صور الإعلان يساوي 1؛ لا تعادل خمس صور خمسة أسعار مستقلة. لم يستخدم نموذج السعر سنة أو ممشى أو اسم السيارة أو الأعلام أو معرفات أو قاعدة إعلانات.',
        '- ضبط C من 10 و100 و1000 داخل 3 طيات GroupKFold من train فقط؛ ε=0.05. إحصاءات الميزات وحدود السعر تتعلم داخل كل طية. صور المجموعة لا تعبر الطيات.',
        '- إحصاءات تطبيع الصور تتعلم من صور الغلاف داخل التدريب، وبالطريقة نفسها للذراعين، لعزل أثر إضافة الزوايا.',
        '- تقييم CV يقارن رأسَي السعر فقط؛ لا يستخدم رأس Gemini المدرب على كل train داخل أي طية. تقييم val يستعمل مزجًا ثابتًا 25% في اللوغاريتم للذراعين، مطابقًا للنسبة التاريخية.',
        '- اختير مرشح واحد لكل ذراع قبل val، ثم قُفل الاختيار. صورة الغلاف محددة قبل النتائج، والتقييم الأساسي لصورة واحدة لكل إعلان. كل الصور تحليل ثانوي بوزن متساو للإعلان؛ لا يجمع صورًا وقت الاستخدام.',
        '- الشروط المسبقة: تحسن MAE بمقدار 5%، MAPE لا يسوء بأكثر من 0.5 نقطة، نسبة ±20% لا تنخفض أكثر من نقطة، وخطأ الشريحة الأغلى لا يزيد أكثر من 5%. فاصل الفرق المقترن يبلغ للمعلومة ولا يشترط استبعاد الصفر وحده.',
        '', '## المقارنة داخل train — اختيار المرشحين', '', '| التدريب | C المختار | MAE ريال | MAPE | ضمن ±20% |', '|---|---:|---:|---:|---:|']
    for arm, label in [('single', 'صورة واحدة'), ('multi', 'كل الصور المؤهلة')]:
        m = cv['selected'][arm]
        out.append(f"| {label} | {m['C']:.0f} | {m['MAE_SAR']:,.2f} | {m['MAPE_percent']:.2f}% | {m['within_20_percent']:.2f}% |")
    ci = cv['paired_bootstrap']['new_minus_reference_MAE_95CI']
    out += ['', f'فاصل 95% لفرق MAE (متعدد الصور − صورة واحدة)، bootstrap مقترن على group_id: [{ci[0]:,.2f}، {ci[1]:,.2f}] ريال. السالب لصالح الجديد. المقارنة استكشافية بعد اختيار C؛ الفاصل لا يصحح تعدد المرشحين.', '',
        '## val الاستكشافية — صورة واحدة ثابتة لكل إعلان', '', '| النسخة | MAE ريال | MAPE | ضمن ±20% |', '|---|---:|---:|---:|']
    for arm, label in [('original', 'النموذج الحالي المقفل'), ('single_control', 'ضبط صورة واحدة داخل train'), ('multi_candidate', 'التدريب بكل الصور')]:
        m = result['primary_val'][arm]
        out.append(f"| {label} | {m['MAE_SAR']:,.2f} | {m['MAPE_percent']:.2f}% | {m['within_20_percent']:.2f}% |")
    ci = result['primary_val']['multi_candidate']['new_minus_reference_MAE_95CI']
    out += ['', f'فاصل 95% لفرق MAE بين الجديد والحالي: [{ci[0]:,.2f}، {ci[1]:,.2f}] ريال. أخذ العينات على group_id، 2000 مرة. النتيجة مشروطة بالمرشح المختار. val سبق استخدامها تاريخيًا، فلا تمثل تحققًا مستقلًا جديدًا.', '',
        '## كل صور val — تحليل ثانوي', '', '| النسخة | MAE بوزن الإعلان | MAPE | ضمن ±20% |', '|---|---:|---:|---:|']
    for arm, label in [('original', 'الحالي'), ('single_control', 'صورة واحدة'), ('multi_candidate', 'متعدد الصور')]:
        m = result['secondary_all_views_equal_listing_weight'][arm]
        out.append(f"| {label} | {m['MAE_SAR']:,.2f} | {m['MAPE_percent']:.2f}% | {m['within_20_percent']:.2f}% |")
    out += ['', '## قرار التجربة والقيود', '', result['decision'], '',
        '- جميع إعدادات المرشحين ونتائج 18 تدريبًا داخل ثلاث طيات في experiments.csv؛ ومجاميع المرشحين الستة في cv_summary.csv. لا أخفي المرشحين الأضعف.',
        '- شروط القرار مفصلة في RESULTS.json؛ شرائح السعر في price_bands.csv؛ لا تخلط تحسن فرع واحد مع نجاح الخطة كاملة.',
        f"- نسبة قبول صور الغلاف بالعتبة القديمة غير المعايرة: {result['acceptance_rate_percent']:.2f}%. لم تضبط العتبة، ولا تمثل الدرجة ثقة في السعر. مقاييس الجدول تشمل الصور المرفوضة أيضًا للتشخيص، ولا تخفيها.",
        '- لم تشغّل أي توقع أو مقياس على test في هذه الجولة. سمح بفك صور صغيرة من الأقسام فقط للتحقق غير المراقب من شبه التكرار؛ لم تستخدم أسعار test. الفحص لا يثبت غياب إعادة نشر نفس السيارة بزوايا مختلفة أو تعديلات كبيرة.',
        '- تجربة تدقيق أولى شديدة التحفظ استبعدت معظم train بسبب تشابه بصمات غير مؤكد، فأُلغيت قبل أي تدريب. استبدلت بتأكيد RGB للمرشحات قبل الاستبعاد، دون الاطلاع على أداء أي نموذج. ملفات التدقيق الأول محفوظة منفصلة.',
        '- لم يبدأ DINOv2، ولم يُستبدل النموذج النشط. نسخة هذه الجولة مرشح تجريبي محفوظ منفصلًا.',
        '- الاختبار الخارجي: لم يختبر بعد. لا معايرة لرفض صور الجوال ولا نطاق سعر فردي معاير. تعليمات الجمع في external_test/image_price_multiview/README_AR.md.',
        '- لا وعد بخطأ 15–20% أو دقة عامة. صورة واحدة قد تشير لفئة وسعر معتاد، لكنها لا تحدد الممشى والحالة الميكانيكية أو السنة والفئة الدقيقة دائمًا. الهدف سعر طلب إعلان تاريخي، لا سعر بيع مضمون.',
        '', '## التشغيل', '', f'مجلد المرشح: `{run}`', '',
        '```powershell', f'.venv/Scripts/python.exe -X utf8 "{run / "runtime.py"}" --bundle "{run}" --image "مسار_صورة_حقيقية"', '```', '',
        'هذا المشغل يتطلب مكتبات بيئة المشروع الحالية، ولا يحتاج جدول الإعلانات وقت التشغيل. لا تعرض السعر الخام للمستخدم إذا كانت الحالة abstained.', '',
        'نجح استدلال صور val وإعادة إنتاج توقعات الرأس ضمن حد التقريب المسجل، وفشل إدخال ملف غير صورة كما يجب، ونجح التحميل من عملية مستقلة. تفاصيل الفرق في runtime_verification.json. نتائج اختبارات الكود تسجّل في test_log.txt إن وجدت.']
    text = '\n'.join(out) + '\n'
    (run / 'REPORT_AR.md').write_text(text, encoding='utf-8')
    (ROOT / 'docs/IMAGE_PRICE_MULTIVIEW_2026-10-23.md').write_text(text, encoding='utf-8')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--run', type=Path)
    a = ap.parse_args(); finish(a.run or ROOT / js(ROOT / 'models/image_price_multiview_work.json')['run_dir'])
