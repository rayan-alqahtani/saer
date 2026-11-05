"""تثبيت إحصاءات val المحفوظة ومراجعة الاختيارات قبل فتح test."""
from improve_common import *
from improve_visual import baseline_probs
import joblib


def verify():
    s=session();guard()
    if (s/'selection_lock.json').exists() or list(s.glob('*/test_once.json')):
        raise ValueError('هذه المراجعة قبل قفل الاختيارات وفتح الاختبار فقط')
    checks=[]
    for task in TASK_INFO:
        cfg=load_json(s/task/'selected.json');rows=compact_rows(task,'val');p=np.load(s/task/'val_probabilities.npy')
        classes=cfg['classes'];y=[r[TASK_INFO[task][1]] for r in rows]
        good=p.shape==(len(rows),len(classes)) and np.isfinite(p).all() and np.all(p>=0) and np.allclose(p.sum(1),1,atol=.0001)
        measured=score(y,p,classes)
        good=bool(good and abs(measured['macro_F1']-cfg['val_macro_F1'])<1e-10)
        old,old_classes=baseline_probs(task,rows)
        if list(old_classes)!=list(classes):raise ValueError('تغير قاموس فئات المهمة')
        class_report(task,'val_before',rows,old,classes)
        class_report(task,'val',rows,p,classes,np.asarray(classes)[old.argmax(1)])
        checks.append({'task':task,'passed':good,'val_macro_F1':measured['macro_F1'],'n':len(rows)})
    rows=compact_rows('price','val');model=joblib.load(s/'price/best_model.joblib')
    p=model.predict(rows);mae=float(np.mean(abs(np.asarray([float(r['sar_price']) for r in rows])-p)))
    cfg=load_json(s/'price/selected.json')
    checks.append({'task':'price','passed':bool(np.isfinite(p).all() and abs(mae-cfg['val_MAE_SAR'])<.00001),'val_MAE_SAR':mae})
    m=load_json(s/'search/val_metrics.json')
    checks.append({'task':'search','passed':not m['same_listing_or_sha_or_group_returned'],'val_top5':m['top5']})
    record={'checked_at':now(),'passed':all(c['passed'] for c in checks),'checks':checks,'test_started':False,'selection_based_on_val_only':True}
    save_json(s/'development_verified.json',record)
    print(json.dumps(record,ensure_ascii=False,indent=2))
    if not record['passed']:raise SystemExit(1)


if __name__=='__main__':verify()
