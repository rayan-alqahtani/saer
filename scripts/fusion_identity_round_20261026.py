"""Final bounded fusion of complementary, already cached foreground/geometry views."""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='2'
from collections import Counter
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits
import foreground_visual_round_20261026 as fg
import geometry_visual_round_20261026 as geo

r=fg.r
OUT=r.ROOT/'models/fusion_identity_20261026_v1'
TASKS=['make','model']
CS=[.1,1.]

def features(rows,task,split):
    old=r.v.load_json(r.v.RUN/task/'selection_lock.json')
    base=r.v.feats(rows,old['kind'],task)
    lookup={z['sha256']:i for i,z in enumerate(r.v.read_csv(fg.OUT/('features_'+split)/'index.csv'))}
    patch=np.load(fg.OUT/('features_'+split)/'patch.npy',mmap_mode='r')[[lookup[z['sha256']] for z in rows]]
    lookup={z['sha256']:i for i,z in enumerate(r.v.read_csv(geo.OUT/('features_'+split)/'index.csv'))}
    letter=np.load(geo.OUT/('features_'+split)/'letterbox.npy',mmap_mode='r')[[lookup[z['sha256']] for z in rows]]
    return np.c_[base,patch,letter]

def fit(x,rows,task,c):
    gc=Counter(z['group_id'] for z in rows);w=np.array([1/gc[z['group_id']] for z in rows]);w/=w.mean()
    sc=StandardScaler().fit(x);m=LogisticRegression(C=c,class_weight='balanced',max_iter=650,random_state=r.SEED)
    m.fit(sc.transform(x),[z[r.TARGET[task]] for z in rows],sample_weight=w)
    current=r.v.load_json(r.v.RUN/task/'result.json')
    return {'model':m,'scaler':sc,'classes':list(m.classes_),'task':task,'C':c,
        'runtime_blend_policy':'half_reference' if current['accepted'] and current['selected_val_option']=='half_reference' else 'direct'}

def main():
    OUT.mkdir(exist_ok=False);fmap=r.v.load_json(r.v.RUN/'folds.json')
    r.save(OUT/'protocol.json',{'seed':r.SEED,'C_grid':CS,'tasks':TASKS,
        'features':'current direct features + foreground DINO patch mean/std + whole-car letterbox DINO CLS',
        'selection':'three original train group folds, equal-listing macro-F1; one selected candidate per task',
        'train_gate':'+.01 macro-F1, accuracy decline<=.01','val_gate':'+.01 macro-F1, positive paired CI, accuracy decline<=.01',
        'runtime':'preserve current ensemble structure, replace only DINO component',
        'test_used':False,'external12_not_for_hyperparameter_selection':True,
        'known_external_inputs_informed_prior_preprocessing_diagnosis':True,
        'data_identity':r.v.load_json(r.ROOT/'data/processed/training_manifest.json')})
    result={};arts={}
    with threadpool_limits(limits=2):
        for task in TASKS:
            rr=r.v.compact_rows(task,'train');cs=sorted({z[r.TARGET[task]] for z in rr});x=features(rr,task,'train')
            ff=np.array([fmap[z['group_id']] for z in rr]);folder=OUT/task;folder.mkdir();scores=[]
            baseline=r.metric(rr,np.load(r.OUT/task/'current_direct_oof.npy'),cs,r.TARGET[task])
            for c in CS:
                p=np.zeros((len(rr),len(cs)),dtype='float32')
                for fold in range(3):
                    a=np.flatnonzero(ff!=fold);b=np.flatnonzero(ff==fold)
                    assert not {rr[i]['group_id'] for i in a}&{rr[i]['group_id'] for i in b}
                    art=fit(x[a],[rr[i] for i in a],task,c);q=art['model'].predict_proba(art['scaler'].transform(x[b]))
                    p[np.ix_(b,[cs.index(v) for v in art['classes']])]=q
                m=r.metric(rr,p,cs,r.TARGET[task]);scores.append({'C':c,**m});np.save(folder/f'fusion_C{c}_oof.npy',p)
                print('FINAL_FUSION_CV',task,c,m,flush=True)
            best=max(scores,key=lambda z:z['macro_f1']);gate=best['macro_f1']-baseline['macro_f1']>=.01 and best['accuracy']>=baseline['accuracy']-.01
            art=fit(x,rr,task,best['C']);arts[task]=art;joblib.dump(art,folder/'head_train_selected.joblib',compress=3)
            result[task]={'reference':baseline,'candidates':scores,'selected':best,'train_gate':bool(gate),'val_evaluated':False,'accepted':False,'test_used':False}
            r.save(folder/'selection_lock.json',result[task])
        r.save(OUT/'diagnostic_selection_lock.json',{'before_external_predictions':True,
            'artifacts_sha256':{t:r.digest(OUT/t/'head_train_selected.joblib') for t in TASKS}})
        for task,decision in result.items():
            if not decision['train_gate']:continue
            rr=r.v.compact_rows(task,'val');art=arts[task];cs=art['classes']
            raw=art['model'].predict_proba(art['scaler'].transform(features(rr,task,'val')))
            new=.5*raw+.5*np.load(r.v.RUN/task/'val_reference.npy') if art['runtime_blend_policy']=='half_reference' else raw
            oldres=r.v.load_json(r.v.RUN/task/'result.json');old=np.load(r.v.RUN/task/('val_selected.npy' if oldres['accepted'] else 'val_reference.npy'))
            before=r.metric(rr,old,cs,r.TARGET[task]);after=r.metric(rr,new,cs,r.TARGET[task]);ci=r.paired_ci(rr,old,new,cs,r.TARGET[task])
            decision.update({'val_evaluated':True,'val_reference':before,'val_candidate':after,'paired':ci,
                'accepted':bool(after['macro_f1']-before['macro_f1']>=.01 and after['accuracy']>=before['accuracy']-.01 and ci['delta_macro_f1_ci95'][0]>0)})
            np.save(OUT/task/'val_candidate.npy',new)
        for task,z in result.items():r.save(OUT/task/'result.json',z)
        r.save(OUT/'summary.json',result);print('FUSION_RESULTS',result,flush=True)

if __name__=='__main__':main()
