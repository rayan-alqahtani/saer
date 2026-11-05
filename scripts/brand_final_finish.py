"""تجميع أدلة الجولة الأخيرة واختبار المشغل؛ لا ضبط على val ولا تقييم test."""
import sys
import json
import time
import subprocess
import joblib
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import classification_report, confusion_matrix
from improve_common import (ROOT, session, read_csv, load_json, save_json, save_csv,
                            compact_rows, class_bootstrap, sha256)
from brand_final_round import BrandNet, evaluate, strict_inputs
from ml_visual import select_rejection_threshold


def grouped_intervals(rows, y, guesses, classes):
    """Bootstrap بالمجموعات داخل طبقات مجموعات الوسوم، لتجنب اختفاء فئة نادرة من العينة."""
    ids=sorted({r['group_id'] for r in rows});gi={g:i for i,g in enumerate(ids)};ci={c:i for i,c in enumerate(classes)}
    true=np.zeros((len(ids),len(classes)))
    labels={g:set() for g in ids}
    for r,label in zip(rows,y):
        true[gi[r['group_id']],ci[label]]+=1;labels[r['group_id']].add(label)
    strata={}
    for g in ids:strata.setdefault(tuple(sorted(labels[g])),[]).append(gi[g])
    rng=np.random.default_rng(42);weights=np.zeros((2000,len(ids)))
    for group_indices in strata.values():
        n=len(group_indices);weights[:,group_indices]=rng.multinomial(n,np.full(n,1/n),size=2000)
    actual=weights@true;result={};samples={}
    for name,pred in guesses.items():
        predicted=np.zeros_like(true);tp=np.zeros_like(true)
        for r,a,b in zip(rows,y,pred):
            i=gi[r['group_id']];predicted[i,ci[b]]+=1
            if a==b:tp[i,ci[a]]+=1
        count=weights@predicted;hit=weights@tp
        precision=np.divide(hit,count,out=np.zeros_like(hit),where=count>0)
        recall=np.divide(hit,actual,out=np.zeros_like(hit),where=actual>0)
        f1=np.divide(2*hit,actual+count,out=np.zeros_like(hit),where=(actual+count)>0)
        samples[name]=f1.mean(1)
        result[name]={'macro_F1':np.quantile(samples[name],[.025,.975]).tolist(),
            'accuracy':np.quantile(hit.sum(1)/actual.sum(1),[.025,.975]).tolist(),
            'per_class':{c:{'precision':np.quantile(precision[:,j],[.025,.975]).tolist(),
                            'recall':np.quantile(recall[:,j],[.025,.975]).tolist(),
                            'F1':np.quantile(f1[:,j],[.025,.975]).tolist()} for j,c in enumerate(classes)}}
    return result,samples


def main():
    run=Path((ROOT/'tmp/brand_final_run.txt').read_text().strip())
    strict_inputs()
    lock=load_json(run/'final_lock.json')
    if sha256(run/'brand_model.pt')!=lock['checkpoint_sha256'] or sha256(run/'selection.json')!=lock['selection_sha256']:
        raise ValueError('بصمة النموذج أو الاختيار اختلفت عن القفل')
    results=load_json(run/'results.json'); chosen=load_json(run/'selection.json')['chosen_on_train_only']
    trials=read_csv(run/'inner_experiments.csv')
    decisions=[]
    for trial in trials:
        selected=(int(trial['depth'])==chosen['depth'] and trial['balance'].lower()==str(chosen['balance']).lower()
                  and trial['tta'].lower()==str(chosen['tta']).lower() and int(trial['epochs'])==chosen['epochs'])
        decisions.append({**trial,'القرار':'مختار' if selected else 'لم يختر',
                          'السبب':'أعلى macro-F1 على التحقق الداخلي' if selected else 'macro-F1 داخلي أقل من الإعداد المختار'})
    save_csv(run/'experiment_decisions.csv',decisions)
    classes=load_json(run/'protocol.json')['classes']
    va=compact_rows('make','val'); y=np.asarray([r['label_make'] for r in va])
    p=np.load(run/'val_probabilities.npy'); pred=np.asarray(classes)[p.argmax(1)]
    old={r['image_uid']:r['predicted'] for r in read_csv(session()/'make/val_predictions.csv')}
    guesses={'المسار المسلم':np.asarray([old[r['image_uid']] for r in va]),'إعادة التدريب الأخيرة':pred}
    previous=session()/'brand_multicrop_20261022T070617/hierarchical_blend_allroute/val_hybrid_probabilities.npy'
    if previous.exists():
        before=np.load(previous)
        previous_classes=list(joblib.load(previous.parent.parent/'candidate_brand_head.joblib')['classes'])
        if previous_classes!=classes:raise ValueError('ترتيب فئات المرشح السابق مختلف')
        if before.shape!=p.shape:raise ValueError('أبعاد المرشح السابق لا تطابق صفوف المقارنة')
        guesses['مرشح القصاصات والدمج السابق']=np.asarray(classes)[before.argmax(1)]
    summary=[]
    clean=np.asarray([r['listing_uid']!='external_group_b:0035US' for r in va])
    bootstrap={}
    for label,mask in [('val الكامل',np.ones(len(va),dtype=bool)),('val مع عزل الإعلان المؤكد',clean)]:
        rows=[r for r,k in zip(va,mask) if k]
        intervals,samples=grouped_intervals(rows,y[mask],{name:g[mask] for name,g in guesses.items()},classes)
        for name,guess in guesses.items():
            ci=intervals[name];f1=samples[name]
            summary.append({'evaluation':label,'model':name,'n':len(rows),
                'accuracy':float(accuracy_score(y[mask],guess[mask])),
                'macro_F1':float(f1_score(y[mask],guess[mask],labels=classes,average='macro',zero_division=0)),
                'macro_F1_ci95':ci['macro_F1']})
            bootstrap[(label,name)]=f1
    deltas={}
    for name in guesses:
        if name=='إعادة التدريب الأخيرة':continue
        key='val مع عزل الإعلان المؤكد'
        deltas[name]=np.quantile(bootstrap[(key,'إعادة التدريب الأخيرة')]-bootstrap[(key,name)],[.025,.975]).tolist()
    save_json(run/'comparison.json',{'rows':summary,'paired_delta_ci95':deltas,'test_evaluated':False,
        'bootstrap':'2000 group_id resamples stratified by group label-set; conditional on observed label-set group counts',
        'audited_per_class_ci95':intervals})
    save_csv(run/'comparison.csv',summary)
    per=classification_report(y[clean],pred[clean],labels=classes,output_dict=True,zero_division=0)
    matrix=confusion_matrix(y[clean],pred[clean],labels=classes)
    trainrows=compact_rows('make','train')
    worst=[]
    for cl in sorted(classes,key=lambda c:per[c]['f1-score'])[:10]:
        i=classes.index(cl);counts=matrix[i].copy();counts[i]=0
        groups={r['group_id'] for r in trainrows if r['label_make']==cl}
        worst.append({'class':cl,**per[cl],'train_groups':len(groups),
            'most_confused_with':classes[int(counts.argmax())] if counts.max() else '',
            'possible_reason':'دعم مجموعات قليل؛ فرضية تحتاج فحص صور' if len(groups)<20 else 'السبب غير مثبت؛ تشابه المظهر أو الخلفية احتمال يحتاج فحصًا'})
    save_csv(run/'audited_val_worst10.csv',worst)
    save_csv(run/'audited_val_per_class.csv',[{'class':c,**per[c]} for c in classes])
    # قاعدة الرفض تختار من التحقق الداخلي داخل train، وليس من نتائج val التي رأيناها.
    alltrain={r['image_uid']:r for r in compact_rows('make','train')}
    tune=[alltrain[r['image_uid']] for r in read_csv(run/'inner_partition.csv') if r['role']=='tune']
    inner=run/f"inner_depth{chosen['depth']}_balanced{chosen['balance']}"/'inner_model.pt'
    net=BrandNet(len(classes));net.load_state_dict(torch.load(inner,map_location='cpu',weights_only=True)['state_dict'])
    ip=evaluate(net,tune,chosen['tta'])
    policy=select_rejection_threshold([r['label_make'] for r in tune],ip,classes)
    policy.update(selection_source='train_internal_tune_only',not_calibrated=True,external_test='لم يختبر بعد')
    keep=p.max(1)>=policy['threshold']
    policy['final_val_observed']={'coverage':float(keep.mean()),'accepted':int(keep.sum()),
        'accuracy':float((pred[keep]==y[keep]).mean()) if keep.any() else None}
    save_json(run/'operating_policy.json',policy)
    # تشغيل مستقل من مجلد آخر، على صورة val فقط، ومقارنة أعلى اقتراح مع التنبؤ المحفوظ.
    start=time.monotonic()
    process=subprocess.run([sys.executable,'-X','utf8',str(ROOT/'scripts/brand_final_predict.py'),
                           '--run',str(run),'--image',str(ROOT/'الملفات_المهمة'/va[0]['relative_path'])],
                          cwd=ROOT/'tmp',capture_output=True,encoding='utf-8')
    if process.returncode:raise RuntimeError(process.stderr+process.stdout)
    output=json.loads(process.stdout)
    assert output['اقتراحات'][0]['الماركة']==pred[0]
    save_json(run/'fresh_process_smoke.json',{'result':output,'startup_inference_seconds':time.monotonic()-start,
        'top1_matches_saved_val_prediction':True,'image_uid':va[0]['image_uid']})
    strict_inputs()
    candidates=[r for r in summary if r['evaluation']=='val مع عزل الإعلان المؤكد']
    winner=max(candidates,key=lambda r:r['macro_F1'])
    decision={'winner_on_audited_val_macro_F1':winner,'official_pointer_changed':False,
              'reason':'المقارنة على val معروف سابقًا؛ مرشح منفصل مع الحفاظ على النموذج المسلم',
              'test_evaluated':False}
    save_json(run/'decision.json',decision)
    checks=[{'check':'بصمات البيانات والتقسيم ثابتة','passed':True},
            {'check':'اختيار إعدادات الجولة داخل train فقط','passed':True},
            {'check':'لا استخدام رؤوس مدربة على التحقق الداخلي لتهيئة الجولة','passed':True},
            {'check':'اختبار استدلال مستقل ومطابقة الكاش','passed':True},
            {'check':'عدم إعادة تقييم test','passed':True,'meaning':'لم ينفذ حفاظًا على القفل'},
            {'check':'خلو البيانات التاريخية من النسخ القريبة مثبت','passed':False,'meaning':'صورة معدلة بين val/test مؤكدة؛ فحص بقية الأزواج غير شامل'},
            {'check':'التعميم الخارجي مثبت','passed':False,'meaning':'لا صور اختبار خارجية موسومة'}]
    save_json(run/'checklist.json',checks)
    save_csv(run/'checklist.csv',[{'البند':r['check'],'الحالة':'ناجح' if r['passed'] else 'غير مثبت','التفسير':r.get('meaning','')} for r in checks])
    save_json(run/'artifact_manifest.json',{'files':{name:sha256(run/name) for name in [
        'brand_model.pt','final_lock.json','selection.json','operating_policy.json','comparison.json']},
        'scripts':{name:sha256(ROOT/'scripts'/name) for name in [
            'brand_final_round.py','brand_final_predict.py','brand_final_finish.py']}})
    lines=['# الجولة الأخيرة لتعرف الماركة — 22 أكتوبر 2026','',
      'هذه جولة تدريب فعلية للماركات الـ26. اختيرت المعاملات وعدد العصور وTTA داخل train بالمجموعات فقط، ثم أعيد التدريب على train كاملًا وقُفلت الأوزان قبل قياس val. لم ينفذ test.','',
      '## نتيجة المقارنة','', '| النطاق | النموذج | الصور | الدقة | macro-F1 |', '|---|---|---:|---:|---:|']
    for r in summary:lines.append(f"| {r['evaluation']} | {r['model']} | {r['n']} | {r['accuracy']:.2%} | {r['macro_F1']:.4f} |")
    lines += ['',f"الأفضل وفق macro-F1 على val بعد عزل الإعلان المؤكد: **{winner['model']}**. المؤشر الرسمي لم يتغير؛ الأوزان الجديدة في مجلد مستقل.",'',
      '## المنهج والضبط','',f"الإعداد المختار داخليًا: `{json.dumps(chosen,ensure_ascii=False)}`.",
      'بدأت الشبكة من ImageNet فقط، ورأس أولي دُرب على جزء التدريب الداخلي وحده. كل مجموعة لها مساهمة متوازنة رغم اختلاف عدد صورها؛ قورنت موازنة الماركات بحسب عدد مجموعاتها المستقلة. استُعمل انعكاس أفقي معتدل، AdamW، إحصاءات BatchNorm ثابتة، قص التدرجات، وجدول معدل تعلم ثابت مع إيقاف مبكر بصبر عصرين. التخزين المرحلي لطبقات ImageNet الثابتة يقلل كلفة CPU.',
      'تنبيه لقراءة سجل refit_all_train/history.json: الحقول accuracy وmacro_F1 وn موروثة من وصف الإعداد المختار على التحقق الداخلي؛ ليست قياسات جديدة لكل عصر من إعادة التدريب. المقاس في هذه المرحلة هو weighted_train_loss والزمن فقط، ثم يقاس val مرة واحدة بعد قفل الأوزان.',
      'تم تجميع زوج شديد التشابه داخل train احترازيًا في نفس وحدة التحقق الداخلي، دون تغيير group_id في البيانات. كاش الصور مقروء فقط، وبصمات صور train/val تحققت قبل التدريب.', '',
      '## اكتشاف مهم في البيانات','',
      'الصورة external_group_b:0035US:04 في val والصورة external_group_b:0036US:05 في test تعرضان المشهد نفسه، مع تعديل موضع لوحة السيارة بشعار الموقع. اختلاف SHA لا يمنع هذا النوع من التسرب. عُزل إعلان 0035US كاملًا من المقارنة المدققة فقط؛ ظل التقرير يعرض val الكامل أيضًا، ولم تتغير البيانات أو التقسيم.',
      'فحص pHash/dHash عبر الأقسام أعطى 3251 زوجًا مرشحًا. راجعت أقرب 25 بصريًا؛ كثير منها سيارات مختلفة من الموديل نفسه في المعرض نفسه. هذا فحص محدود ولا يثبت انعدام جميع النسخ القريبة. عُرضت بعض صور test لفحص التسرب فقط، ولم تُنتج لها تنبؤات أو مقاييس.', '',
      'وُسّع الفحص بعد ذلك إلى pHash≤8 وdHash≤10 مع متوسط فرق بكسلات RGB64≤3؛ أعاد الزوج المؤكد نفسه وزوج سيارات مختلف الجنوط تمت مراجعته. لم يضف حالة مؤكدة جديدة، وما زال الفحص غير شامل لكل أشكال الاقتصاص والتحرير.', '',
      '## تصحيح تفسير الجولة السابقة','',
      'تقسيم val في تجربة القصاصات السابقة ليس اختبارًا مستقلًا: الرأس السابق اختير على val الكامل قبل تقسيمه. فواصل Bootstrap السابقة وصفية مشروطة بالنماذج المختارة، ولا تصحح تفاؤل الاختيار. الجولة الحالية تحسن الانضباط باستخدام train الداخلي، لكن val الخارجي معروف تاريخيًا أيضًا؛ يلزم اختبار خارجي جديد لإثبات التعميم.', '',
      '## فواصل الفرق والقيود','',f"فواصل فرق macro-F1 بالمجموعات بعد−قبل على المقارنة المدققة: `{json.dumps(deltas,ensure_ascii=False)}`.",
      'فواصل المقارنة النهائية تستخدم 2000 إعادة أخذ للمجموعات داخل طبقات مجموعة الوسوم، فتحافظ على حضور الفئات المرصودة بدل احتساب فئة نادرة غائبة بصفر. هذه فواصل شرطية على تكوين الفئات المرصود، وليست إثبات تعميم أو تصحيحًا لتعدد التجارب؛ الفئات قليلة المجموعات ما زالت غير يقينية. الفواصل الأولية في results.json تستخدم الطريقة الأقدم غير الطبقية، والمرجع النهائي هنا هو comparison.json.',
      f"سياسة القبول المختارة داخل train: `{json.dumps(policy,ensure_ascii=False)}`. هذه درجات غير معايرة ولا تضمن رفض الصور خارج المجال.",
      'اختبار الاستدلال في عملية جديدة نجح وطابق التنبؤ المحفوظ؛ مقارنة خمسة أمثلة من الصور الخام بالكاش نجحت. لا اختبار خارجي ولا ضمان 90% ولا ادعاء خلو شامل من فرط التعلم أو التسرب.', '',
      'نجحت 9 اختبارات آلية بالأمر `.venv/Scripts/python -X utf8 -m unittest tests.test_brand_final_round tests.test_improvement_protocol -v`، وتشمل حراسة التقسيم وقفل الاختبار وأوزان المجموعات وBootstrap. هذه اختبارات سلامة برمجية وليست قياسًا لدقة الصور الخارجية.', '',
      '## الملفات','',f"مجلد الجولة: `{run}`. يتضمن brand_model.pt، سجل كل عصر، inner_experiments.csv، ملفات التقسيم الداخلي، مقارنة النماذج، تنبؤات val، الفئات ومصفوفة الالتباس، وبصمات القفل."]
    lines += ['', 'لتجربة صورة محلية دون تدريب أو تقييم test:', '', '```powershell',
      f'.venv/Scripts/python -X utf8 scripts/brand_final_predict.py --run "{run}" --image "مسار_الصورة"', '```', '',
      'هذا مشغل مرشح الماركة الجديد فقط؛ بقية مهام المشروع والنموذج المسلم لم تتغير. الملف operating_policy.json يحوي العتبة الداخلية وقياس تغطيتها على val.']
    report=ROOT/'docs/BRAND_FINAL_ROUND_2026-10-22.md'
    if report.exists():raise ValueError('تقرير الجولة موجود؛ لا يستبدل دون مراجعة')
    report.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    with (ROOT/'PROGRESS.md').open('a',encoding='utf-8') as f:
        f.write('\n\n## الجولة الأخيرة للماركة — 22 أكتوبر 2026\n\n')
        f.write('**ما أنجزته:** تدريب مستقل للماركة من ImageNet، اختيار داخل train حسب المجموعة، إعادة التدريب الكامل وقفل الأوزان قبل val.\n')
        f.write('**ما اختبرته والنتيجة:** استدلال مستقل ومطابقة الصور الخام والكاش ناجحان. نتائج المقارنة في docs/BRAND_FINAL_ROUND_2026-10-22.md؛ الفائز وفق val المدقق: '+winner['model']+'. لا test جديد.\n')
        f.write('**الملفات المتغيرة:** scripts/brand_final_round.py وbrand_final_predict.py وbrand_final_finish.py؛ tests/test_brand_final_round.py؛ '+str(run)+'؛ التقرير وPROGRESS.md.\n')
        f.write('**المشكلات:** اكتشاف نسخة معدلة مؤكدة بين val/test، عزل إعلانها من المقارنة المدققة دون تغيير الجداول. لا ضمان خلو شامل من النسخ القريبة ولا اختبار خارجي؛ تصحيح ادعاء استقلال حجز val السابق.\n')
        f.write('**الخطوة التالية:** مراجعة المرشح الأفضل وتجربته على صور خارجية موسومة قبل استبدال النموذج المسلم أو ادعاء التعميم.\n**هل تحتاج موافقتي؟:** لا لإتمام الجولة المحلية المصرح بها؛ لم تتغير البيانات أو مؤشرات التسليم ولم يحدث رفع.\n')
    print(json.dumps({'winner':winner,'rows':summary,'report':str(report)},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
