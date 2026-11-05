"""تقييم صور خارجية حقيقية موسومة يدويًا؛ لا صور أو نتائج مصطنعة."""
from improve_common import *
from improve_visual import predict_selected,make_prediction
import argparse
from sklearn.metrics import classification_report

EXTERNAL=ROOT/'external_test'


def setup():
    (EXTERNAL/'images').mkdir(parents=True,exist_ok=True)
    path=EXTERNAL/'labels.csv'
    if not path.exists():save_csv(path,[],fields=['image_path','group_id','source_type','manual_label_confirmed','make','rec_label','color','body_type'])
    text='''<div dir="rtl" lang="ar">

# اختبار حقيقي من خارج مصدر التدريب

ضع 20–30 صورة سيارات فعلية من جوالك أو من مصدر آخر لديك إذن لاستخدامه داخل images/. لا تضع صور غير مصرح باستخدامها أو لقطات من بيانات المشروع. لا نولد صورًا أو وسومًا لاختبار الدقة.

املأ labels.csv يدويًا، صفًا لكل صورة:

- image_path: مسار نسبي مثل images/اسم_صورتك.jpg.
- group_id: معرف السيارة الحقيقية؛ كل صور السيارة نفسها تحمل المعرف نفسه، ولا تعطي كل زاوية معرف سيارة جديدًا.
- source_type: phone أو other_source.
- manual_label_confirmed: True بعد التحقق البشري من الوسوم.
- make: اسم الماركة مطابق لقائمة الفئات.
- rec_label: الماركة|الموديل كما في قاموس الـ59؛ اتركه فارغًا إن لم تكن متأكدًا، أو اكتب الاسم الحقيقي إن كان خارج النطاق؛ سيظهر غير مدعوم دون تخمين.
- color وbody_type: من الفئات المعتمدة إذا تحقق منها الإنسان؛ اترك المجهول فارغًا. body_type هو شكل السيارة، وليس وصف الضرر.

الفئات الدقيقة متاحة في selected.json لكل مهمة داخل مجلد التحسين المشار إليه في models/improvement_current.json. ستظهر الفئات غير المدعومة وأعدادها ونسبة التغطية منفصلة عن الدقة. لا تغير الوسوم لرفع الدقة.

التشغيل من C:/CAR:

```powershell
& './.venv/Scripts/python.exe' -X utf8 ./scripts/improve_external.py evaluate
```

يقارن SHA بالصور الأصلية ويستبعد التطابق المطابق من الاختبار الخارجي، ويحسب accuracy وmacro-F1 وprecision/recall/F1 لكل فئة، وفاصل 95% بإعادة أخذ عينات group_id. يحفظ النتائج في results/ ويطبع الملخص. المجموعة الصغيرة استكشافية، ولا تبرهن دقة عامة. لا تستخدم نتائجها لتحسين النموذج ثم تسمي المجموعة نفسها اختبارًا مستقلًا.

ملف labels.csv المبدئي يحوي العناوين فقط. لا توجد صور أو دقة خارجية قبل إضافة صور حقيقية ووسومها اليدوية.

</div>
'''
    save_bytes(EXTERNAL/'README.md',text.encode('utf-8'))


def evaluate_external():
    guard();setup();labels=read_csv(EXTERNAL/'labels.csv')
    if not labels:
        result={'status':'لم يختبر بعد','reason_ar':'لا توجد صور ووسوم خارجية حقيقية؛ لم أحسب دقة','n':0}
        save_json(EXTERNAL/'results/status.json',result);print(json.dumps(result,ensure_ascii=False));return
    existing={r['sha256'] for r in read_csv(PROCESSED/'images.csv')};usable=[];excluded=[];seen={}
    for n,r in enumerate(labels):
        path=(EXTERNAL/r['image_path']).resolve()
        if not path.is_relative_to(EXTERNAL.resolve()) or not path.is_file():raise ValueError('مسار صورة غير صالح: '+r['image_path'])
        if not r['group_id'] or r['manual_label_confirmed'].lower()!='true':raise ValueError('يلزم group_id وتأكيد بشري لكل صورة')
        if r['source_type'] not in ('phone','other_source'):raise ValueError('حدد مصدرًا خارجيًا حقيقيًا')
        digest=sha256(path)
        if digest in existing:excluded.append({'image_path':r['image_path'],'reason':'SHA موجود في بيانات المشروع'});continue
        if digest in seen:
            if seen[digest]!=r['group_id']:raise ValueError('صورة مطابقة وضعت في مجموعتين خارجيتين مختلفتين')
            excluded.append({'image_path':r['image_path'],'reason':'صورة خارجية مطابقة مكررة؛ لا تعد عينة جديدة'});continue
        seen[digest]=r['group_id']
        usable.append({**r,'external_path':str(path),'image_uid':'external:'+r['image_path'],'listing_uid':r['group_id'],
            'sha256':digest,'label_make':r['make'],'label_color_std':r['color'],'label_body_type_std':r['body_type']})
    summary={'n_input':len(labels),'n_distinct_sha':len({r['sha256'] for r in usable}),'excluded_exact_overlap':excluded,
             'scope_ar':'اختبار خارجي استكشافي؛ لا ضمان لتعميم السوق','tasks':{}}
    for task,(flag,target) in TASK_INFO.items():
        classes=load_json(session()/task/'selected.json')['classes'];known=[r for r in usable if r[target] in classes]
        unknown=[r for r in usable if r[target] and r[target] not in classes]
        if not known:
            summary['tasks'][task]={'status':'لم يختبر','known_target_images':0,'unsupported_target_images':len(unknown)};continue
        p,classes=make_prediction(known) if task=='make' else predict_selected(task,known)
        y=np.asarray([r[target] for r in known]);pred=np.asarray(classes)[p.argmax(1)]
        ci,_,_=class_bootstrap(known,y,pred,classes)
        per=classification_report(y,pred,labels=classes,output_dict=True,zero_division=0)
        save_csv(EXTERNAL/f'results/{task}_per_class.csv',[{'class':cl,**per[cl]} for cl in classes])
        summary['tasks'][task]={**score(y,p,classes),'ci95':ci,'unsupported_target_images':len(unknown),
            'unlabeled_images':sum(not r[target] for r in usable),
            'macro_F1_observed_true_classes':float(f1_score(y,pred,labels=sorted(set(y)),average='macro',zero_division=0)),
            'macro_F1_scope_note':'macro_F1 وفاصله للفئات المدعومة كاملة، والقياس الإضافي للفئات الحقيقية الموجودة في العينة فقط؛ لا تخلط التعريفين'}
    save_json(EXTERNAL/'results/metrics.json',summary);print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('command',choices=['setup','evaluate']);a=ap.parse_args()
    if a.command=='setup':setup()
    else:evaluate_external()
