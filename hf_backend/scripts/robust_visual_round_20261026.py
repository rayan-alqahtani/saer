"""Bounded experiments on the current frozen DINO/ResNet feature caches.

Train-group CV chooses configurations; at most one new candidate per task is
compared on historical val. Test is never requested. Published models stay intact.
"""
from __future__ import annotations
import os
for _name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[_name] = '2'
import json, csv, time, warnings, hashlib, platform
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
import joblib
import sklearn
from scipy.special import softmax
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import f1_score, accuracy_score
from threadpoolctl import threadpool_limits
import visual_fusion_round as v

ROOT = v.ROOT
OUT = ROOT / 'models/robust_visual_20261026_v1'
SEED = 20261026
TASKS = ['make', 'model', 'color', 'body']
TARGET = v.old.TARGET

def save(p, x):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(x, ensure_ascii=False, indent=2), encoding='utf-8')

def table(p, rows):
    if not rows: return
    with p.open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def norm(x):
    return np.asarray(x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-10), dtype='float32')

class Cache:
    def __init__(self):
        self.index = {r['sha256']: i for i,r in enumerate(v.read_csv(v.RUN/'features/index.csv'))}
        self.full = np.load(v.RUN/'features/full.npy', mmap_mode='r')
        self.crop = np.load(v.RUN/'features/crop.npy', mmap_mode='r')
    def dino(self, rows):
        ii = [self.index[r['sha256']] for r in rows]
        return norm(self.full[ii]), norm(self.crop[ii])

CACHE = None

def features(rows, task, cfg):
    rep = cfg['rep']
    if rep == 'current': return v.feats(rows, cfg['kind'], task)
    if rep == 'color_only': return v.feats(rows, 'color_only', task)
    full, crop = CACHE.dino(rows)
    if rep in ('aug', 'prototype'): return full, crop
    if rep == 'full': return full
    if rep == 'mean': return norm(full + crop)
    if rep == 'joined': return np.column_stack([full, crop]) / np.sqrt(2)
    raise ValueError(rep)

def weights(rows, target, exponent):
    gc = Counter(r['group_id'] for r in rows)
    per_class = defaultdict(set)
    for r in rows: per_class[r[target]].add(r['group_id'])
    cc = {c: len(g) for c,g in per_class.items()}
    cw = {c: (sum(cc.values()) / (len(cc)*n))**exponent for c,n in cc.items()}
    w = np.array([cw[r[target]]/gc[r['group_id']] for r in rows], dtype='float64')
    return w / w.mean()

def fit(rows, task, cfg):
    target = TARGET[task]
    x = features(rows, task, cfg)
    y = np.array([r[target] for r in rows])
    w = weights(rows, target, cfg.get('exponent', .5))
    scaler = None
    if cfg['rep'] in ('aug', 'prototype'):
        full,crop = x
        x = np.concatenate([full,crop]); y = np.tile(y,2); w = np.tile(w/2,2)
        if cfg['rep'] == 'prototype':
            by = defaultdict(list)
            for i,r in enumerate(rows): by[r['listing_uid']].append(i)
            xx=[]; yy=[]; ww=[]
            for ids in by.values():
                labels = {rows[i][target] for i in ids}
                if len(labels)!=1: raise ValueError('listing label conflict')
                xx.append(norm(np.mean(np.r_[full[ids],crop[ids]],axis=0)[None])[0])
                yy.append(rows[ids[0]][target]); ww.append(float(w[ids].sum()*2))
            w *= .7
            x = np.r_[x,np.asarray(xx)]; y = np.r_[y,yy]
            w = np.r_[w,.3*np.asarray(ww)]
    if cfg.get('scale'):
        scaler = StandardScaler().fit(x, sample_weight=w)
        x = scaler.transform(x)
    if cfg['head']=='reference':
        # Exact fitting settings of the published DINO direct component.
        gc=Counter(r['group_id'] for r in rows)
        w=np.array([1/gc[r['group_id']] for r in rows]);w/=w.mean()
        scaler=StandardScaler().fit(x)
        model=LogisticRegression(C=cfg['C'], class_weight='balanced', max_iter=600, random_state=20261025)
        x=scaler.transform(x)
    elif cfg['head']=='logit':
        model=LogisticRegression(C=cfg['C'], max_iter=700, random_state=SEED)
    elif cfg['head']=='ridge':
        model=RidgeClassifier(alpha=cfg['alpha'])
    elif cfg['head']=='trees':
        model=ExtraTreesClassifier(n_estimators=200, min_samples_leaf=cfg['leaf'],
            max_features=.7, n_jobs=2, random_state=SEED)
    else: raise ValueError(cfg)
    with warnings.catch_warnings(record=True) as ww:
        warnings.simplefilter('always');model.fit(x,y,sample_weight=w)
    return {'config':cfg,'model':model,'scaler':scaler,'task':task,
        'classes':list(model.classes_),'warnings':[str(z.message) for z in ww]}

def predict(art, rows):
    x=features(rows,art['task'],art['config'])
    xs=x if isinstance(x,tuple) else (x,)
    pp=[]
    for z in xs:
        if art['scaler'] is not None: z=art['scaler'].transform(z)
        m=art['model']
        if hasattr(m,'predict_proba'): p=m.predict_proba(z)
        else:
            a=m.decision_function(z)
            if a.ndim==1: a=np.c_[-a,a]
            p=softmax(a,axis=1)
        pp.append(p)
    return np.mean(pp,axis=0)

def metric(rows, p, classes, target):
    y=np.array([r[target] for r in rows]);pred=np.asarray(classes)[p.argmax(1)]
    lc=Counter(r['listing_uid'] for r in rows);w=np.array([1/lc[r['listing_uid']] for r in rows])
    first={}
    for i,r in sorted(enumerate(rows),key=lambda z:z[1]['image_uid']): first.setdefault(r['listing_uid'],i)
    ii=list(first.values())
    return {'macro_f1':float(f1_score(y,pred,labels=classes,average='macro',sample_weight=w,zero_division=0)),
        'accuracy':float(accuracy_score(y,pred,sample_weight=w)),
        'image_macro_f1':float(f1_score(y,pred,labels=classes,average='macro',zero_division=0)),
        'image_accuracy':float(accuracy_score(y,pred)),
        'fixed_view_macro_f1':float(f1_score(y[ii],pred[ii],labels=classes,average='macro',zero_division=0)),
        'images':len(rows),'listings':len(first)}

def paired_ci(rows,a,b,classes,target):
    y=np.array([r[target] for r in rows]);pa=np.array(classes)[a.argmax(1)];pb=np.array(classes)[b.argmax(1)]
    groups=sorted({r['group_id'] for r in rows});gi={g:i for i,g in enumerate(groups)};cl={c:i for i,c in enumerate(classes)}
    lc=Counter(r['listing_uid'] for r in rows)
    mats=[np.zeros((len(groups),len(classes))) for _ in range(5)]
    true,ap,bp,ah,bh=mats
    for i,r in enumerate(rows):
        g=gi[r['group_id']];w=1/lc[r['listing_uid']];true[g,cl[y[i]]]+=w
        ap[g,cl[pa[i]]]+=w;bp[g,cl[pb[i]]]+=w;ah[g,cl[y[i]]]+=w*(pa[i]==y[i]);bh[g,cl[y[i]]]+=w*(pb[i]==y[i])
    rng=np.random.default_rng(SEED);draw=rng.multinomial(len(groups),np.full(len(groups),1/len(groups)),size=2000)
    t=draw@true
    dena=t+draw@ap;denb=t+draw@bp
    fa=np.divide(2*(draw@ah),dena,out=np.zeros_like(dena),where=dena>0).mean(1)
    fb=np.divide(2*(draw@bh),denb,out=np.zeros_like(denb),where=denb>0).mean(1)
    return {'delta_macro_f1_ci95':np.quantile(fb-fa,[.025,.975]).tolist(),
        'unit':'group_id','replicates':2000,'conditional_on_selected_configuration':True}

def configs(task):
    selected=v.load_json(v.RUN/task/'selection_lock.json')
    out=[{'name':'current_direct','rep':'current','kind':selected['kind'],'head':'reference','C':selected['C']}]
    if task!='color':
        for rep,c in [('full',10.),('full',100.),('mean',10.),('mean',100.),('joined',10.),
                      ('aug',10.),('aug',100.),('prototype',10.)]:
            out.append({'name':f'{rep}_logit_C{c:g}','rep':rep,'head':'logit','C':c,'exponent':.5})
    else:
        for c in [.01,.1,1.]:
            out.append({'name':f'color_logit_C{c:g}','rep':'current','kind':'both','head':'logit',
                        'C':c,'exponent':.5,'scale':True})
        for leaf in [3,10]:
            out.append({'name':f'paint_trees_leaf{leaf}','rep':'color_only','head':'trees','leaf':leaf,'exponent':.5})
    for alpha in [10.,100.]:
        out.append({'name':f'ridge_alpha{alpha:g}','rep':'current','kind':selected['kind'],
                    'head':'ridge','alpha':alpha,'scale':True,'exponent':.5})
    return out

def check_protected(protected):
    for name,h in protected.items():
        if digest(ROOT/name)!=h: raise AssertionError('Protected input changed: '+name)

def run_task(task):
    target=TARGET[task];tr=v.compact_rows(task,'train');classes=sorted({r[target] for r in tr})
    folds=v.load_json(v.RUN/'folds.json');ff=np.array([folds[r['group_id']] for r in tr])
    folder=OUT/task;folder.mkdir()
    trial=[];probs={}
    for cfg in configs(task):
        started=time.perf_counter();p=np.zeros((len(tr),len(classes)),dtype='float32');logs=[]
        for fold in range(3):
            a=np.flatnonzero(ff!=fold);b=np.flatnonzero(ff==fold)
            aa=[tr[i] for i in a];bb=[tr[i] for i in b]
            assert not {r['group_id'] for r in aa}&{r['group_id'] for r in bb}
            art=fit(aa,task,cfg);q=predict(art,bb)
            ix=[classes.index(c) for c in art['classes']];p[np.ix_(b,ix)]=q
            logs.extend(art['warnings'])
        m=metric(tr,p,classes,target)
        trial.append({'name':cfg['name'],**m,'seconds':time.perf_counter()-started,'warnings':len(logs)})
        probs[cfg['name']]=p;np.save(folder/(cfg['name']+'_oof.npy'),p)
        save(folder/(cfg['name']+'_config.json'),{'config':cfg,'warnings':logs})
        table(folder/'train_configs.csv',trial)
        print(task,cfg['name'],json.dumps(m),flush=True)
    ref=trial[0]
    best=max(trial[1:],key=lambda r:r['macro_f1'])
    eligible=(best['macro_f1']-ref['macro_f1']>=.01 and best['accuracy']>=ref['accuracy']-.01)
    result={'train_reference_direct_component':ref,'train_selected':best,'train_gate':bool(eligible),
        'selected_using':'train group CV only; selected CV scores exploratory',
        'test_used':False,'val_evaluated':False,'accepted':False}
    save(folder/'selection_lock.json',result)
    if eligible:
        cfg=next(c for c in configs(task) if c['name']==best['name']);art=fit(tr,task,cfg)
        joblib.dump(art,folder/'head.joblib',compress=3)
        # Only the previously train-selected candidate is evaluated here.
        va=v.compact_rows(task,'val');new=predict(art,va)
        oldrows=v.read_csv(v.RUN/task/'val_predictions.csv')
        assert [r['image_uid'] for r in oldrows]==[r['image_uid'] for r in va]
        oldresult=v.load_json(v.RUN/task/'result.json')
        old=np.load(v.RUN/task/('val_selected.npy' if oldresult['accepted'] else 'val_reference.npy'))
        before=metric(va,old,classes,target);after=metric(va,new,classes,target);ci=paired_ci(va,old,new,classes,target)
        result.update({'val_evaluated':True,'val_reference_runtime':before,'val_candidate':after,
            'paired':ci,'accepted':bool(after['macro_f1']-before['macro_f1']>=.01
            and after['accuracy']>=before['accuracy']-.01 and ci['delta_macro_f1_ci95'][0]>0)})
        np.save(folder/'val_candidate.npy',new)
        table(folder/'val_predictions.csv',[{'image_uid':r['image_uid'],'listing_uid':r['listing_uid'],
            'group_id':r['group_id'],'actual':r[target],'reference':classes[old[i].argmax()],
            'candidate':classes[new[i].argmax()]} for i,r in enumerate(va)])
    save(folder/'result.json',result)
    print('RESULT',task,json.dumps(result),flush=True)
    return result

def main():
    global CACHE
    OUT.mkdir(exist_ok=False)
    c,_=v.old.compare_inputs()
    assert not c['critical'] and all(x['authorized_match'] for x in c['files'].values())
    protected={str(p.relative_to(ROOT)):digest(p) for p in (ROOT/'models').glob('*current.json')}
    for p in [ROOT/'الملفات_المهمة/ربط_الصور.csv',ROOT/'الملفات_المهمة/الداتا_المحدثة.csv',
              v.RUN/'features/full.npy',v.RUN/'features/crop.npy']:
        protected[str(p.relative_to(ROOT))]=digest(p)
    save(OUT/'protocol.json',{'seed':SEED,'candidate_configs':{t:configs(t) for t in TASKS},
        'data':c,'identity':v.load_json(ROOT/'data/processed/training_manifest.json'),
        'protected':protected,'primary_metric':'macro-F1 over all single-image predictions, equal total weight per listing',
        'secondary':'unweighted image metrics and first fixed image per listing; no multi-image inference',
        'CV':'existing 3 train group folds; train-only selection, no independent accuracy claim after selection',
        'val':'historically reused; at most one candidate per task, only after +.01 train macro-F1 and no >.01 accuracy loss',
        'acceptance':'val macro-F1 +.01, paired 95% bootstrap lower >0, accuracy drop <=.01; development only',
        'test':'not requested or inferred','external12':'not used for selection','backbones':'frozen DINOv2 and ImageNet ResNet18',
        'environment':{'python':platform.python_version(),'numpy':np.__version__,'sklearn':sklearn.__version__}})
    CACHE=Cache();result={}
    with threadpool_limits(limits=2):
        for task in TASKS:
            result[task]=run_task(task);check_protected(protected)
    save(OUT/'summary.json',result)
    save(OUT/'integrity.json',{'protected_sha_match':True,'test_inference':False,'active_pointers_changed':False})

if __name__=='__main__':main()
