"""Nested train-group selection of complementary frozen CLIP + existing features."""
from __future__ import annotations
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='2'
import json,time,warnings
from collections import Counter
from pathlib import Path
import numpy as np
import joblib
from scipy.special import softmax
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits
import clip_identity_round_20261027 as clip
import fusion_identity_round_20261026 as fusion

r=clip.r;OUT=clip.OUT
CS=[.1,1.]
GAMMAS=[0.,.25]


def load_features(rows,task,split):
    idx={z['image_uid']:i for i,z in enumerate(r.v.read_csv(OUT/'features/index.csv'))}
    ii=[idx[z['image_uid']] for z in rows]
    full=np.load(OUT/'features/full.npy',mmap_mode='r')[ii]
    crop=np.load(OUT/'features/crop.npy',mmap_mode='r')[ii]
    base=fusion.features(rows,task,split) if task=='make' else r.v.feats(rows,r.v.load_json(r.v.RUN/task/'selection_lock.json')['kind'],task)
    x=np.c_[base,full,crop]
    text=np.load(OUT/f'{task}_text.npy')
    scale=r.v.load_json(OUT/'extraction_complete.json')['text_logit_scale']
    semantic=clip.norm(full+crop)@text.T*scale
    return base,x,semantic


def fit_predict(x,rows,xq,c,classes):
    groupcounts=Counter(z['group_id'] for z in rows)
    w=np.array([1/groupcounts[z['group_id']] for z in rows]);w/=w.mean()
    scaler=StandardScaler().fit(x)
    model=LogisticRegression(C=c,class_weight='balanced',max_iter=650,random_state=20261026)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        model.fit(scaler.transform(x),[z['_target'] for z in rows],sample_weight=w)
    messages=[str(z.message) for z in caught]
    with (OUT/'fit_events.jsonl').open('a',encoding='utf-8') as stream:
        stream.write(json.dumps({'fit_images':len(rows),'fit_groups':len(groupcounts),
            'query_images':len(xq),'dimensions':x.shape[1],'C':c,
            'iterations':int(model.n_iter_.max()),'warnings':messages})+'\n')
    p=model.predict_proba(scaler.transform(xq))
    assert np.isfinite(p).all() and np.allclose(p.sum(axis=1),1.,atol=1e-5)
    q=np.zeros((len(xq),len(classes)),dtype='float32');q[:,[classes.index(c) for c in model.classes_]]=p
    return q,{'scaler':scaler,'model':model,'classes':list(model.classes_),'C':c,
             'fit_warnings':messages,'iterations':int(model.n_iter_.max())}


def with_text(p,semantic,gamma):
    ps=softmax(semantic,axis=1)
    strength=np.clip(1.+(ps*np.log(np.maximum(ps,1e-12))).sum(1)/np.log(ps.shape[1]),0,1)
    return softmax(np.log(np.maximum(p,1e-10))+gamma*strength[:,None]*(semantic-semantic.mean(1,keepdims=True)),axis=1)


def run_task(task):
    dest=OUT/task
    dest.mkdir(exist_ok=False)
    rows=r.v.compact_rows(task,'train');target=r.TARGET[task]
    for z in rows:z['_target']=z[target]
    classes=sorted({z[target] for z in rows})
    base,x,semantic=load_features(rows,task,'train')
    assert len(x)==len(rows) and semantic.shape==(len(rows),len(classes))
    fmap=r.v.load_json(r.v.RUN/'folds.json');ff=np.array([fmap[z['group_id']] for z in rows])
    reference=np.zeros((len(rows),len(classes)),dtype='float32');selected=reference.copy()
    candidates={(c,g):reference.copy() for c in CS for g in GAMMAS}
    logs=[]
    for fold in range(3):
        started=time.perf_counter();aa=np.flatnonzero(ff!=fold);bb=np.flatnonzero(ff==fold)
        ar=[rows[i] for i in aa];gg=np.array([z['group_id'] for z in ar])
        assert not set(gg)&{rows[i]['group_id'] for i in bb}
        assert not {z['sha256'] for z in ar}&{rows[i]['sha256'] for i in bb}
        inner={c:np.zeros((len(ar),len(classes)),dtype='float32') for c in CS}
        for j,(fit,hold) in enumerate(GroupKFold(3).split(gg,groups=gg)):
            assert not set(gg[fit])&set(gg[hold])
            for c in CS:
                p,_=fit_predict(x[aa[fit]],[ar[i] for i in fit],x[aa[hold]],c,classes)
                inner[c][hold]=p
            print('CLIP_INNER',task,fold,j,flush=True)
        scores=[]
        for c in CS:
            for gamma in GAMMAS:
                scores.append({'C':c,'text_weight':gamma,**r.metric(ar,with_text(inner[c],semantic[aa],gamma),classes,target)})
        choice=max(scores,key=lambda z:(z['macro_f1'],z['accuracy'],-z['text_weight'],-z['C']))
        ref,_=fit_predict(base[aa],ar,base[bb],1.,classes);reference[bb]=ref
        for c in CS:
            p,_=fit_predict(x[aa],ar,x[bb],c,classes)
            for gamma in GAMMAS:
                candidates[(c,gamma)][bb]=with_text(p,semantic[bb],gamma)
        selected[bb]=candidates[(choice['C'],choice['text_weight'])][bb]
        logs.append({'outer_fold':fold,'chosen':choice,'trials':scores,'seconds':time.perf_counter()-started})
        r.save(dest/f'outer_{fold}.json',logs[-1]);print('CLIP_OUTER',task,fold,'seconds',round(logs[-1]['seconds'],1),flush=True)
    before=r.metric(rows,reference,classes,target);after=r.metric(rows,selected,classes,target)
    ci=r.paired_ci(rows,reference,selected,classes,target)
    gate=after['macro_f1']-before['macro_f1']>=.01 and after['accuracy']>=before['accuracy']-.005 and ci['delta_macro_f1_ci95'][0]>0
    scores=[{'C':c,'text_weight':gamma,**r.metric(rows,p,classes,target)} for (c,gamma),p in candidates.items()]
    final=max(scores,key=lambda z:(z['macro_f1'],z['accuracy'],-z['text_weight'],-z['C']))
    r.save(dest/'selection_lock.json',{'before_val':True,'choice':final,'train_gate':bool(gate),'configs':scores})
    np.savez_compressed(dest/'nested_oof.npz',baseline=reference,candidate=selected)
    result={'task':task,'train_reference_raw_component':before,'train_nested_candidate':after,
            'train_paired':ci,'train_gate':bool(gate),'final_selection':final,
            'val_evaluated':False,'accepted':False,'test_read':False,'external_used_for_selection':False}
    if gate:
        va=r.v.compact_rows(task,'val');vb,vx,vs=load_features(va,task,'val')
        saved=r.v.read_csv(r.v.RUN/task/'val_predictions.csv')
        assert [z['image_uid'] for z in saved]==[z['image_uid'] for z in va]
        p,art=fit_predict(x,rows,vx,final['C'],classes)
        assert list(art['model'].classes_)==classes
        art.update({'task':task,'text_weight':final['text_weight'],'feature_kind':'existing_direct_features_plus_clip_full_crop',
                    'classes':classes,'trained_on':'train only','source_protocol_sha256':r.digest(OUT/'head_protocol.json')})
        joblib.dump(art,dest/'head.joblib',compress=3)
        new=with_text(p,vs,final['text_weight'])
        old=np.load(fusion.OUT/'make/val_candidate.npy') if task=='make' else np.load(r.v.RUN/'model/val_selected.npy')
        before=r.metric(va,old,classes,target);after=r.metric(va,new,classes,target);ci=r.paired_ci(va,old,new,classes,target)
        result.update({'val_evaluated':True,'val_reference_current_full_runtime':before,'val_candidate':after,'val_paired':ci,
            'accepted':bool(after['macro_f1']-before['macro_f1']>=.01 and after['accuracy']>=before['accuracy']-.005 and ci['delta_macro_f1_ci95'][0]>0)})
        np.savez_compressed(dest/'val_predictions.npz',baseline=old,candidate=new)
        r.table(dest/'val_predictions.csv',[{'image_uid':z['image_uid'],'listing_uid':z['listing_uid'],'group_id':z['group_id'],
                    'actual':z[target],'baseline':classes[old[i].argmax()],'candidate':classes[new[i].argmax()]} for i,z in enumerate(va)])
    r.save(dest/'result.json',result)
    print('CLIP_RESULT',task,json.dumps(result),flush=True)
    return result


def main():
    if not (OUT/'extraction_complete.json').exists():raise RuntimeError('Feature extraction not complete')
    if (OUT/'head_protocol.json').exists():raise FileExistsError('Refusing a repeat experiment')
    r.save(OUT/'head_protocol.json',{'candidate_C':CS,'text_prior_weights':GAMMAS,'seed':20261026,
        'primary':'all single-image predictions with equal listing weight','train_selection':'nested3x3 group cross-fitting',
        'baseline':'make raw newest fused component, model direct DINO component; all refit in same group folds',
        'val_baseline':'current full latest make/runtime model blends, not an older historical test artifact',
        'selection_bias_note':'raw components and representations previously chosen on historical val; nested comparison conditional on prior choices',
        'no_test':True,'no_external_labels_for_selection':True,'code_sha256':r.digest(Path(__file__))})
    with threadpool_limits(limits=2):
        result={t:run_task(t) for t in ('make','model')}
    r.check_protected(r.v.load_json(OUT/'protocol.json')['protected'])
    r.save(OUT/'results.json',result)


if __name__=='__main__':main()
