"""قفل جميع اختيارات val ثم حدث اختبار نهائي واحد لكل مهمة."""
from improve_common import *
from improve_visual import baseline_probs,predict_selected,route_probs
from improve_price import price_metrics
from improve_search import evaluate
from ml_visual import select_rejection_threshold
import joblib,argparse


def lock():
    guard();s=session();names=[]
    for task in ['make','model','color','body','price','search']:
        p=s/task/'selected.json'
        if not p.exists():raise ValueError('اختيار غير مكتمل: '+task)
        names.append(p)
    for task in ['model','color','body']:
        cfg=load_json(s/task/'selected.json')
        if cfg['kind']=='tail':names.append(s/cfg['checkpoint'])
    names += [s/'price/best_model.joblib',s/'price/median_baseline.joblib']
    if load_json(s/'make/selected.json').get('independent_source')=='adapted_head':names.append(s/'make/adapted_head.joblib')
    if load_json(s/'search/selected.json')['kind']=='finetuned_cosine':
        names += [s/'search/finetuned_features.npy',s/'search/finetuned_index.csv']
    with (s/'selection_lock.json').open('x',encoding='utf-8') as f:
        json.dump({'locked_at':now(),'artifacts_sha256':{str(p.relative_to(s)):sha256(p) for p in names},
            'source_code_sha256':{str(p.relative_to(ROOT)):sha256(p) for p in (ROOT/'scripts').glob('improve_*.py')},
            'test_metrics_not_used_for_selection':True},f,ensure_ascii=False,indent=2)
    print('FINAL SELECTION LOCKED',flush=True)


def old_predictions(task,rows):
    saved={r['image_uid']:r for r in read_csv(original(task)/'test_image_predictions.csv')}
    return [saved[r['image_uid']]['predicted'] for r in rows]


def finish_lock(path,metrics_file):
    state=load_json(path);state.update(state='completed',completed_at=now(),metrics_file=str(metrics_file.relative_to(session())),metrics_sha256=sha256(metrics_file))
    save_json(path,state)


def final_classification():
    s=session();make_lock=acquire_test('make');model_lock=acquire_test('model')
    make_rows=compact_rows('make','test');model_rows=compact_rows('model','test')
    union={r['sha256']:r for r in make_rows+model_rows};shared=list(union.values());lookup={r['sha256']:i for i,r in enumerate(shared)}
    # forward واحد للمصنف الجديد على اتحاد صور المهمتين، ثم مشاركة النتيجة.
    cfg=load_json(s/'make/selected.json')
    if cfg.get('independent_source')=='adapted_head':
        from improve_make_head import head_probs
        new_model,model_classes,features=predict_selected('model',shared,True)
        head=joblib.load(s/'make/adapted_head.joblib');pm=head_probs(features[[lookup[r['sha256']] for r in make_rows]],head);make_classes=head['classes']
    else:
        pm,make_classes=baseline_probs('make',make_rows);new_model,model_classes=predict_selected('model',shared)
    # احفظ نواتج forward قبل حساب الإحصاءات؛ إعادة حساب CI لا تستلزم تشغيل النموذج.
    np.save(s/'model/shared_final_probabilities.npy',new_model)
    np.save(s/'make/independent_final_probabilities.npy',pm)
    save_csv(s/'model/shared_final_index.csv',[{'sha256':r['sha256'],'image_uid':r['image_uid']} for r in shared])
    if cfg['kind']=='independent':makep=pm
    else:
        if cfg['model_source']=='improved':pc=new_model[[lookup[r['sha256']] for r in make_rows]]
        else:pc,model_classes=baseline_probs('model',make_rows)
        makep=route_probs(pm,pc,model_classes,make_classes,cfg['threshold'],cfg['method'])
        oracle=route_probs(pm,pc,model_classes,make_classes,0.,method=cfg['method'],mask=[r['usable_recognition']=='True' for r in make_rows])
        class_report('make','test_metadata_route',make_rows,oracle,make_classes,old_predictions('make',make_rows))
    modelp=new_model[[lookup[r['sha256']] for r in model_rows]]
    np.save(s/'make/test_probabilities.npy',makep);np.save(s/'model/test_probabilities.npy',modelp)
    class_report('make','test',make_rows,makep,make_classes,old_predictions('make',make_rows))
    class_report('model','test',model_rows,modelp,model_classes,old_predictions('model',model_rows))
    finish_lock(make_lock,s/'make/test_metrics.json');finish_lock(model_lock,s/'model/test_metrics.json')
    print('TEST FINAL make/model completed ONCE',flush=True)
    for task in ['color','body']:
        path=acquire_test(task);rows=compact_rows(task,'test');p,classes=predict_selected(task,rows)
        np.save(s/task/'test_probabilities.npy',p);class_report(task,'test',rows,p,classes,old_predictions(task,rows))
        finish_lock(path,s/task/'test_metrics.json');print('TEST FINAL',task,'completed ONCE',flush=True)


def final_price():
    path=acquire_test('price');s=session();folder=s/'price';rows=compact_rows('price','test')
    model=joblib.load(folder/'best_model.joblib');baseline=joblib.load(folder/'median_baseline.joblib')
    p=model.predict(rows);pb=baseline.predict(rows)
    np.savez(folder/'test_raw_predictions.npz',prediction=p,median_baseline=pb)
    new,nb=price_metrics(rows,p);base,bb=price_metrics(rows,pb)
    old_rows=[r for r in read_csv(original('price')/'test_predictions.csv') if r['candidate']==load_json(original('price')/'metrics.json')['selected_candidate']]
    old={r['listing_uid']:float(r['predicted_sar']) for r in old_rows};op=np.asarray([old[r['listing_uid']] for r in rows])
    previous,ob=price_metrics(rows,op)
    new.update(before=previous,median_baseline=base,paired_delta_MAE_ci95=np.quantile(nb-ob,[.025,.975]).tolist())
    save_json(folder/'test_metrics.json',new)
    save_csv(folder/'test_predictions.csv',[{'listing_uid':r['listing_uid'],'group_id':r['group_id'],'make':r['make'],'model_std':r['model_std'],
        'actual':r['sar_price'],'before':float(o),'after':float(v),'median_baseline':float(b),'absolute_error':abs(float(r['sar_price'])-v)} for r,o,v,b in zip(rows,op,p,pb)])
    grouped=defaultdict(list)
    for i,r in enumerate(rows):grouped[r['make']+'|'+r['model_std']].append(i)
    segments=[]
    for label,ids in grouped.items():
        m=price_metrics([rows[i] for i in ids],p[ids])[0]
        segments.append({'class':label,'n':len(ids),'MAE_SAR':m['MAE_SAR'],'MAPE_percent':m['MAPE_percent'],
            'possible_reason_ar':'قلة الدعم أو اختلاف تجهيز/حالة غير ممثلة؛ فرضية لا سبب مثبت. لا precision/recall للانحدار.'})
    save_csv(folder/'test_worst10.csv',sorted(segments,key=lambda r:r['MAE_SAR'],reverse=True)[:10]);save_csv(folder/'test_per_model.csv',segments)
    finish_lock(path,folder/'test_metrics.json');print('TEST FINAL price completed ONCE',flush=True)


def final_search():
    path=acquire_test('search');cfg=load_json(session()/'search/selected.json')
    reference=evaluate('test',mode='frozen_resnet_cosine',prefix='test_before')
    if cfg['kind']=='frozen_resnet_cosine':
        for suffix in ['metrics.json','predictions.csv','per_class.csv','worst10.csv']:
            save_bytes(session()/'search'/('test_'+suffix),(session()/'search'/('test_before_'+suffix)).read_bytes())
    else:evaluate('test',mode=cfg['kind'])
    finish_lock(path,session()/'search/test_metrics.json');print('TEST FINAL search completed ONCE',flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('command',choices=['lock','evaluate']);a=ap.parse_args()
    if a.command=='lock':lock()
    else:
        final_classification();final_price();final_search();guard()
