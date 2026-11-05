"""Bounded retrieval-residual diagnostic, inspired by Tip-Adapter (not replica).

Fit/tune/holdout all live inside the original train split. Old test is not read.
"""
from __future__ import annotations
import os
for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[name]='2'
import json,time,csv
from pathlib import Path
from collections import Counter,defaultdict
import numpy as np
import joblib
from sklearn.model_selection import GroupShuffleSplit,GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits
import train_clip_identity_heads_20261027 as ch

r=ch.r;ROOT=r.ROOT;OUT=ROOT/'outputs/cache_identity_adapter_20261028_v1'
BETAS=[5.5,20.,80.];MIXES=[0.,.15,.3,.5];SEED=20261028


def fit_head(x,rows,xq,task,classes,semantic):
    counts=Counter(z['group_id'] for z in rows);w=np.array([1/counts[z['group_id']] for z in rows]);w/=w.mean()
    scaler=StandardScaler().fit(x);c=.1 if task=='make' else 1.
    model=LogisticRegression(C=c,class_weight='balanced',max_iter=650,random_state=20261026)
    model.fit(scaler.transform(x),[z[r.TARGET[task]] for z in rows],sample_weight=w)
    raw=model.predict_proba(scaler.transform(xq));p=np.zeros((len(xq),len(classes)),dtype='float32')
    p[:,[classes.index(c) for c in model.classes_]]=raw
    if task=='make':p=ch.with_text(p,semantic,.25)
    return p,{'scaler':scaler,'model':model,'classes':list(model.classes_),'task':task,'C':c}


def retrieval_features(rows):
    ci={z['image_uid']:i for i,z in enumerate(r.v.read_csv(ch.OUT/'features/index.csv'))}
    di={z['sha256']:i for i,z in enumerate(r.v.read_csv(r.v.RUN/'features/index.csv'))}
    ix=[ci[z['image_uid']] for z in rows];jx=[di[z['sha256']] for z in rows]
    cf=np.load(ch.OUT/'features/full.npy',mmap_mode='r')[ix];cc=np.load(ch.OUT/'features/crop.npy',mmap_mode='r')[ix]
    df=np.load(r.v.RUN/'features/full.npy',mmap_mode='r')[jx];dc=np.load(r.v.RUN/'features/crop.npy',mmap_mode='r')[jx]
    return np.c_[ch.clip.norm(cf+cc),ch.clip.norm(df+dc)].astype('float32')/np.sqrt(2)


def cache_probabilities(gallery,query,rows,classes,task):
    # A group contributes total one vote within each class; each class has unit
    # cache mass. This differs deliberately from original few-shot Tip-Adapter.
    target=r.TARGET[task]
    counts=Counter((z['group_id'],z[target]) for z in rows)
    class_groups={c:len({z['group_id'] for z in rows if z[target]==c}) for c in classes}
    weights=np.zeros((len(rows),len(classes)),dtype='float32')
    for i,z in enumerate(rows):
        c=z[target];weights[i,classes.index(c)]=1./counts[(z['group_id'],c)]/class_groups[c]
    result={b:np.zeros((len(query),len(classes)),dtype='float32') for b in BETAS}
    for st in range(0,len(query),128):
        sim=np.clip(query[st:st+128]@gallery.T,-1,1)
        for b in BETAS:
            score=np.exp((sim-1)*b)@weights
            result[b][st:st+128]=score/np.maximum(score.sum(1,keepdims=True),1e-30)
    return result,weights


def assert_disjoint(a,b):
    for key in ('group_id','listing_uid','sha256'):
        assert not {z[key] for z in a}&{z[key] for z in b},key


def run_task(task,tune_groups,hold_groups):
    folder=OUT/task;folder.mkdir()
    rows=r.v.compact_rows(task,'train');classes=sorted({z[r.TARGET[task]] for z in rows})
    base,combined,semantic=ch.load_features(rows,task,'train')
    x=combined if task=='make' else base
    q=retrieval_features(rows)
    tune=np.array([i for i,z in enumerate(rows) if z['group_id'] in tune_groups])
    hold=np.array([i for i,z in enumerate(rows) if z['group_id'] in hold_groups])
    tr=[rows[i] for i in tune];hr=[rows[i] for i in hold]
    assert len(tune)+len(hold)==len(rows)
    assert_disjoint(tr,hr)
    pred=np.zeros((len(tr),len(classes)),dtype='float32')
    caches={b:pred.copy() for b in BETAS};groups=np.array([z['group_id'] for z in tr])
    for fold,(a,b) in enumerate(GroupKFold(3).split(groups,groups=groups)):
        ar=[tr[i] for i in a];br=[tr[i] for i in b];assert_disjoint(ar,br)
        pred[b],_=fit_head(x[tune[a]],ar,x[tune[b]],task,classes,semantic[tune[b]])
        result,_=cache_probabilities(q[tune[a]],q[tune[b]],ar,classes,task)
        for beta in BETAS:caches[beta][b]=result[beta]
        print('CACHE_INNER',task,fold,flush=True)
    trials=[]
    for beta in BETAS:
        for weight in MIXES:
            p=(1-weight)*pred+weight*caches[beta]
            trials.append({'beta':beta,'cache_weight':weight,**r.metric(tr,p,classes,r.TARGET[task])})
    choice=max(trials,key=lambda z:(z['macro_f1'],z['accuracy'],-z['cache_weight'],-z['beta']))
    r.save(folder/'selection_lock.json',{'before_holdout_predictions':True,'choice':choice,'trials':trials})
    old,art=fit_head(x[tune],tr,x[hold],task,classes,semantic[hold])
    cache,_=cache_probabilities(q[tune],q[hold],tr,classes,task)
    new=(1-choice['cache_weight'])*old+choice['cache_weight']*cache[choice['beta']]
    before=r.metric(hr,old,classes,r.TARGET[task]);after=r.metric(hr,new,classes,r.TARGET[task])
    ci=r.paired_ci(hr,old,new,classes,r.TARGET[task])
    gate=after['macro_f1']-before['macro_f1']>=.01 and after['accuracy']>=before['accuracy']-.005 and ci['delta_macro_f1_ci95'][0]>0
    result={'baseline_internal_holdout':before,'candidate_internal_holdout':after,'paired':ci,
        'chosen_on_tune_only':choice,'holdout_gate':bool(gate),'val_evaluated':False,'accepted':False,
        'original_train_only':True,'test_read':False,'external_images_used':False,
        'note':'internal holdout is conditional on previous representation/head choices; not independent external accuracy'}
    np.savez_compressed(folder/'holdout_predictions.npz',baseline=old,candidate=new)
    r.table(folder/'holdout_predictions.csv',[{'image_uid':z['image_uid'],'listing_uid':z['listing_uid'],'group_id':z['group_id'],
       'actual':z[r.TARGET[task]],'baseline':classes[old[i].argmax()],'candidate':classes[new[i].argmax()]} for i,z in enumerate(hr)])
    if gate:
        va=r.v.compact_rows(task,'val');vb,vx,vs=ch.load_features(va,task,'val');vq=retrieval_features(va)
        newbase,art=fit_head(x,rows,vx if task=='make' else vb,task,classes,vs)
        rc,weights=cache_probabilities(q,vq,rows,classes,task)
        new=(1-choice['cache_weight'])*newbase+choice['cache_weight']*rc[choice['beta']]
        old=np.load(ch.OUT/'make/val_predictions.npz')['candidate'] if task=='make' else np.load(r.v.RUN/'model/val_selected.npy')
        b=r.metric(va,old,classes,r.TARGET[task]);a=r.metric(va,new,classes,r.TARGET[task]);ci=r.paired_ci(va,old,new,classes,r.TARGET[task])
        accepted=a['macro_f1']-b['macro_f1']>=.01 and a['accuracy']>=b['accuracy']-.005 and ci['delta_macro_f1_ci95'][0]>0
        result.update({'val_evaluated':True,'val_reference_current':b,'val_candidate':a,'val_paired':ci,'accepted':bool(accepted)})
        art.update({'classes':classes,'policy':choice});joblib.dump(art,folder/'head.joblib',compress=3)
        np.savez_compressed(folder/'gallery.npz',features=q,class_weights=weights)
        r.table(folder/'gallery_index.csv',[{'image_uid':z['image_uid'],'listing_uid':z['listing_uid'],'group_id':z['group_id']} for z in rows])
        np.savez_compressed(folder/'val_predictions.npz',baseline=old,candidate=new)
    r.save(folder/'result.json',result)
    print('CACHE_RESULT',task,json.dumps(result),flush=True)
    return result


def error_audit():
    rows=r.v.compact_rows('model','train');classes=sorted({z['rec_label'] for z in rows})
    p=np.load(r.OUT/'model/current_direct_oof.npy');assert len(p)==len(rows)
    # Raw component OOF errors provide priorities; no historical test errors used.
    listing={z['listing_uid']:z for z in r.v.read_csv(ROOT/'data/processed/listings.csv') if z['split']=='train'}
    pairs=defaultdict(set);by=defaultdict(list)
    for z,pr in zip(rows,p):
        pred=classes[pr.argmax()];by[z['rec_label']].append((z,pred))
        if pred!=z['rec_label']:pairs[(z['rec_label'],pred)].add(z['group_id'])
    stats=[]
    for c,rr in by.items():
        cnt=Counter(z['listing_uid'] for z,_ in rr);w=np.array([1/cnt[z['listing_uid']] for z,_ in rr])
        years=defaultdict(set)
        for z,_ in rr:years[listing[z['listing_uid']].get('year','')].add(z['group_id'])
        stats.append({'class':c,'train_images':len(rr),'train_listings':len(cnt),'train_groups':len({z['group_id'] for z,_ in rr}),
            'oof_accuracy_equal_listing':float(np.average([p==c for _,p in rr],weights=w)),
            'year_groups_json':json.dumps({year:len(gs) for year,gs in sorted(years.items())},ensure_ascii=False),
            'year_count':len(years),'years_are_not_generation_labels':True})
    r.table(OUT/'training_model_priorities.csv',sorted(stats,key=lambda z:(z['oof_accuracy_equal_listing'],z['train_groups'])))
    confusions=[{'actual':a,'predicted':b,'independent_error_groups':len(gs)} for (a,b),gs in pairs.items()]
    confusions.sort(key=lambda z:z['independent_error_groups'],reverse=True);r.table(OUT/'training_confusions.csv',confusions)
    r.save(OUT/'data_needs.json',{'supported_model_classes':len(classes),
        'classes_below_10_groups':[z['class'] for z in stats if z['train_groups']<10],
        'classes_below_20_groups':[z['class'] for z in stats if z['train_groups']<20],
        'classes_with_one_recorded_year':[z['class'] for z in stats if z['year_count']==1],
        'worst_oof_components':sorted(stats,key=lambda z:z['oof_accuracy_equal_listing'])[:8],
        'top_confusions':confusions[:10],
        'limits':'priorities from train OOF raw component, not latest full runtime, not causal proof; years do not identify generations'})


def main():
    OUT.mkdir(exist_ok=False)
    rows=r.v.compact_rows('make','train');groups=sorted({z['group_id'] for z in rows})
    a,b=next(GroupShuffleSplit(n_splits=1,test_size=.3,random_state=SEED).split(groups,groups=groups))
    tune={groups[i] for i in a};hold={groups[i] for i in b}
    protected={str(p.relative_to(ROOT)):r.digest(p) for p in (ROOT/'models').glob('*current.json')}
    for p in [ROOT/'data/processed/training_manifest.json',ch.OUT/'results.json']:
        protected[str(p.relative_to(ROOT))]=r.digest(p)
    r.save(OUT/'protocol.json',{'seed':SEED,'original_split':'train only; original train/val/test unchanged',
       'internal_groups':{'tuning':len(tune),'holdout':len(hold)},'internal_fold':'3 GroupKFold within tuning groups',
       'gallery':'fitting groups only; group- and class-normalized affinity exp(beta*(cosine-1))',
       'features':'normalized mean full/crop CLIP concatenated normalized mean full/crop DINO; divided sqrt(2)',
       'beta':BETAS,'cache_mix':MIXES,'holdout_gate':'macro-F1 +.01; accuracy decline<=.005; paired group CI lower>0',
       'baseline':'newest CLIP make C=.1 text=.25; DINO direct model C=1; refit only on fitting groups',
       'method_status':'inspired by Tip-Adapter; differs in features, balancing, and probability mixture; not original paper replication',
       'source':'https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136950487.pdf',
       'no_test':True,'no_external_for_selection':True,'protected':protected})
    with threadpool_limits(limits=2):
        error_audit();result={t:run_task(t,tune,hold) for t in ('make','model')}
    r.check_protected(protected);r.save(OUT/'results.json',result)


if __name__=='__main__':main()
