"""بروتوكول تحسين منفصل: بيانات ثابتة، اختيار بالتحقق، وقفل اختبار أحادي."""
from ml_common import *
from day21_dataset_contract import split_fingerprint
from pathlib import Path
from collections import Counter,defaultdict
import csv,json,hashlib,shutil
import numpy as np
from sklearn.metrics import accuracy_score,f1_score,classification_report,confusion_matrix
from scipy.sparse import csr_matrix

POINTER=ROOT/'models/improvement_current.json'
TASK_INFO={'make':('usable_make_recognition','label_make'),'model':('usable_recognition','rec_label'),
           'color':('usable_color','label_color_std'),'body':('usable_body_type','label_body_type_std')}
AUTHORIZED={'listings.csv':'4fe1f6f463b4cce771b733bf0e0c69637da3c51b01af730c36489db673e90e38',
            'images.csv':'c1698a1d1786020d55269f8c5286845f3085c3f537b8db4418269bd58bc54e25'}


def session():
    explicit=os.environ.get('CAR_IMPROVEMENT_RUN')
    if explicit:
        path=Path(explicit).resolve()
        if not path.is_relative_to((ROOT/'models').resolve()):raise ValueError('جلسة التحسين خارج models')
        return path
    return ROOT/load_json(POINTER)['run_dir']


def compare_inputs():
    result={'checked_at':now(),'files':{},'differences':[]};tables={}
    for filename,key,idkey in [('الداتا_المحدثة.csv','listings.csv','listing_uid'),('ربط_الصور.csv','images.csv','image_uid')]:
        p=ROOT/'الملفات_المهمة'/filename;rows=read_csv(p);tables[key]=rows
        result['files'][key]={'path':str(p),'sha256':sha256(p),'authorized_sha256':AUTHORIZED[key],
            'rows':len(rows),'columns':len(rows[0]),'authorized_match':sha256(p)==AUTHORIZED[key]}
        canonical={r[idkey]:r for r in read_csv(PROCESSED/key)}
        columns=[idkey,'listing_uid','split','group_id']+sorted(k for k in rows[0] if k.startswith('usable_'))
        columns=list(dict.fromkeys(columns))
        for r in rows:
            for col in columns:
                old=canonical.get(r[idkey],{}).get(col)
                if old!=r.get(col):result['differences'].append({'table':key,'id':r[idkey],'column':col,'compact':r.get(col),'canonical':old})
        missing=set(canonical)-{r[idkey] for r in rows}
        for uid in missing:result['differences'].append({'table':key,'id':uid,'column':'id_missing_compact'})
    expected=load_json(PROCESSED/'training_manifest.json')
    result['split_fingerprint']=split_fingerprint(tables['listings.csv'])
    result['split_matches_manifest']=result['split_fingerprint']==expected['split_fingerprint']
    groups=defaultdict(set)
    for r in tables['listings.csv']:groups[r['group_id']].add(r['split'])
    result['cross_split_groups']={g:sorted(s) for g,s in groups.items() if len(s)>1}
    result['critical']=not result['split_matches_manifest'] or bool(result['cross_split_groups']) or any(d['column'] in ('split','group_id') for d in result['differences'])
    return result,tables


def guard():
    result,_=compare_inputs();s=session()
    # سجل مستقل لكل استدعاء؛ لا تتنافس عمليات التدريب والاستدلال على tmp واحد.
    tag=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')
    save_json(s/'input_checks'/f'{tag}_{os.getpid()}.json',result)
    if result['critical']:raise ValueError('توقف: اختلاف split أو group_id أو تسرب مجموعة')
    protocol=load_json(s/'protocol.json')
    if result['split_fingerprint']!=protocol['split_fingerprint']:
        raise ValueError('توقف: تغير التقسيم أو المجموعات عن بداية جلسة التحسين')
    for path,digest in protocol['original_models_sha256'].items():
        if sha256(ROOT/path)!=digest:raise ValueError('تغيرت نسخة نموذج مرجعي محفوظة: '+path)
    return result


def compact_rows(task,split=None):
    table='listings.csv' if task=='price' else 'images.csv'
    name='الداتا_المحدثة.csv' if table=='listings.csv' else 'ربط_الصور.csv'
    rows=read_csv(ROOT/'الملفات_المهمة'/name)
    flag='usable_price' if task=='price' else 'usable_visual_search' if task=='search' else TASK_INFO[task][0]
    rows=[r for r in rows if r[flag]=='True' and (split is None or r['split']==split)]
    if task!='price':
        hashes=[r['sha256'] for r in rows]
        if len(set(hashes))!=len(hashes):raise ValueError('SHA متكرر في المهمة')
    return rows


def original(task):return ROOT/load_json(ROOT/f'models/{task}_current.json')['run_dir']


def init():
    check,tables=compare_inputs()
    if check['critical']:raise ValueError('قسم أو مجموعة غير مطابقين')
    folder=ROOT/'models'/('improvement_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f'))
    folder.mkdir(parents=True,exist_ok=False)
    frozen={}
    for task in ['make','model','color','body','price','search','embeddings','delivery']:
        ptr=ROOT/f'models/{task}_current.json';frozen[str(ptr.relative_to(ROOT))]=sha256(ptr)
        for p in (ROOT/load_json(ptr)['run_dir']).rglob('*'):
            if p.is_file():frozen[str(p.relative_to(ROOT))]=sha256(p)
    identity=load_json(PROCESSED/'training_manifest.json')
    save_json(folder/'protocol.json',{'created_at':now(),'seed':42,'bootstrap_repetitions':2000,
        'input_files':check['files'],'split_fingerprint':identity['split_fingerprint'],
        'original_models_sha256':frozen,'prior_test_already_exposed':True,
        'test_policy':'one recorded final evaluation event per task; no tuning after final lock',
        'selection':'single-image val macro-F1; price val MAE with train-only GroupKFold diagnostics',
        'make_routing':'metadata route reported separately from deployable confidence route',
        'data_policy':'no edits; input compact file SHA logged, canonical manifest unchanged',
        'bootstrap_unit':'group_id; same resamples for paired before/after',
        'architecture':'ResNet18 pretrained local; gradual layer4 then layer3; BN running statistics frozen',
        'dashboard_limits':['لا وسوم ضرر أو أثره السعري','فاصل الثقة لمقياس الأداء ليس فاصل تنبؤ فردي']})
    save_json(folder/'input_check_initial.json',check)
    save_json(POINTER,{'run_dir':str(folder.relative_to(ROOT))})
    for task in ['make','model','color','body','price','search']:(folder/task).mkdir()
    shutil.copy2(ROOT/'data/audit/day21/improvement_baseline.json',folder/'baseline_snapshot.json')
    print('SESSION',folder,'INPUT_DIFFS',len(check['differences']),'DISK_FREE_GB',round(shutil.disk_usage(ROOT).free/1e9,1),flush=True)


def experiment(task,name,change,before,after,accepted,extra=None):
    s=session()
    if (s/'selection_lock.json').exists():raise ValueError('التجارب مقفلة بعد تثبيت الاختيار النهائي')
    path=s/'experiments.csv';rows=read_csv(path) if path.exists() else []
    files=compare_inputs()[0]['files']
    rows.append({'time_utc':now(),'task':task,'experiment':name,'single_change':change,
        'selection_split':'val','before':before,'after':after,'accepted':accepted,
        'listings_sha256':files['listings.csv']['sha256'],'images_sha256':files['images.csv']['sha256'],
        'split_fingerprint':load_json(s/'protocol.json')['split_fingerprint'],
        'details':json.dumps(extra or {},ensure_ascii=False)})
    save_csv(path,rows)
    print(task,name,'val',before,'->',after,'KEEP' if accepted else 'REJECT',flush=True)


def score(y,p,classes):
    pred=np.asarray(classes)[np.argmax(p,axis=1)]
    return {'n':len(y),'accuracy':float(accuracy_score(y,pred)),
            'macro_F1':float(f1_score(y,pred,labels=classes,average='macro',zero_division=0))}


def class_bootstrap(rows,y,pred,classes,reps=2000):
    groups=sorted({r['group_id'] for r in rows});gmap={g:i for i,g in enumerate(groups)};cmap={c:i for i,c in enumerate(classes)}
    true=np.zeros((len(groups),len(classes)));guess=np.zeros_like(true);tp=np.zeros_like(true)
    for r,a,b in zip(rows,y,pred):
        g=gmap[r['group_id']];true[g,cmap[a]]+=1;guess[g,cmap[b]]+=1
        if a==b:tp[g,cmap[a]]+=1
    rng=np.random.default_rng(42);weights=rng.multinomial(len(groups),np.full(len(groups),1/len(groups)),size=reps)
    # معظم الخانات صفر؛ الضرب المتناثر يحافظ على العينات نفسها ويقلل حساب CI.
    t=np.asarray((csr_matrix(true).T@weights.T).T)
    p=np.asarray((csr_matrix(guess).T@weights.T).T)
    hit=np.asarray((csr_matrix(tp).T@weights.T).T)
    acc=hit.sum(1)/t.sum(1);class_f1=np.divide(2*hit,t+p,out=np.zeros_like(hit),where=(t+p)>0);f1=class_f1.mean(1)
    precision=np.divide(hit,p,out=np.zeros_like(hit),where=p>0);recall=np.divide(hit,t,out=np.zeros_like(hit),where=t>0)
    class_ci={cl:{'precision':np.quantile(precision[:,j],[.025,.975]).tolist(),
                  'recall':np.quantile(recall[:,j],[.025,.975]).tolist(),
                  'F1':np.quantile(class_f1[:,j],[.025,.975]).tolist()} for j,cl in enumerate(classes)}
    return {'accuracy':np.quantile(acc,[.025,.975]).tolist(),'macro_F1':np.quantile(f1,[.025,.975]).tolist(),
            'bootstrap_unit':'group_id','groups':len(groups),'repetitions':reps,'fixed_class_set':True,
            'per_class':class_ci,'absent_class_resample_policy':'zero_division=0; small support intervals are unstable'},acc,f1


def class_report(task,split,rows,p,classes,before=None):
    target=TASK_INFO[task][1];y=np.asarray([r[target] for r in rows]);pred=np.asarray(classes)[p.argmax(1)]
    folder=session()/task;ci,accboot,f1boot=class_bootstrap(rows,y,pred,classes)
    per=classification_report(y,pred,labels=classes,output_dict=True,zero_division=0)
    save_csv(folder/f'{split}_per_class.csv',[{'class':cl,**per[cl],
        **{f'{metric}_ci95_{side}':ci['per_class'][cl][metric][index] for metric in ['precision','recall','F1'] for index,side in enumerate(['low','high'])}} for cl in classes])
    matrix=confusion_matrix(y,pred,labels=classes)
    save_json(folder/f'{split}_confusion.json',{'classes':classes,'matrix':matrix.tolist()})
    tr=compact_rows(task,'train');support=Counter(r[target] for r in tr)
    worst=[]
    for cl in sorted(classes,key=lambda c:per[c]['f1-score'])[:min(10,len(classes))]:
        i=classes.index(cl);counts=matrix[i].copy();counts[i]=0;j=int(counts.argmax())
        worst.append({'class':cl,'precision':per[cl]['precision'],'recall':per[cl]['recall'],'F1':per[cl]['f1-score'],
            'eval_support':per[cl]['support'],'train_images':support[cl],
            'most_confused_with':classes[j] if counts[j]>0 else '',
            'possible_reason_ar':'دعم تقييم قليل وعدم يقين مرتفع' if per[cl]['support']<10 else 'تشابه المظهر أو الإضاءة والخلفية؛ فرضية من المصفوفة وليست سببًا مثبتًا'})
    save_csv(folder/f'{split}_worst10.csv',worst)
    result={**score(y,p,classes),'ci95':ci}
    majority=Counter(r[target] for r in tr).most_common(1)[0][0]
    result['train_majority_baseline']={'class':majority,'accuracy':float(np.mean(y==majority))}
    if before is not None:
        old=np.asarray(before);oldci,oa,of=class_bootstrap(rows,y,old,classes)
        result['before']={'accuracy':float(accuracy_score(y,old)),
            'macro_F1':float(f1_score(y,old,labels=classes,average='macro',zero_division=0)),'ci95':oldci}
        result['paired_delta_ci95']={'accuracy':np.quantile(accboot-oa,[.025,.975]).tolist(),'macro_F1':np.quantile(f1boot-of,[.025,.975]).tolist()}
    save_json(folder/f'{split}_metrics.json',result)
    save_csv(folder/f'{split}_predictions.csv',[{'image_uid':r['image_uid'],'listing_uid':r['listing_uid'],'group_id':r['group_id'],
        'actual':a,'predicted':b,'score':float(pr.max())} for r,a,b,pr in zip(rows,y,pred,p)])
    return result


def acquire_test(task):
    guard();s=session();lock=load_json(s/'selection_lock.json')
    for name,digest in lock['artifacts_sha256'].items():
        if sha256(s/name)!=digest:raise ValueError('تغير اختيار النموذج بعد القفل')
    path=s/task/'test_once.json'
    with path.open('x',encoding='utf-8') as f:json.dump({'started_at':now(),'state':'started','evaluation_count':1,'selection_lock_sha256':sha256(s/'selection_lock.json')},f)
    return path


if __name__=='__main__':init()
