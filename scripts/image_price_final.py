"""تحسين السعر الصوري: بصمات ثابتة، اختيار val، ثم اختبار نهائي وحيد."""
import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
import shutil
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '4')
os.environ.setdefault('MKL_NUM_THREADS', '4')
import numpy as np
import torch
from sklearn.svm import SVR
from threadpoolctl import threadpool_limits

ROOT = next((p for p in [Path(__file__).resolve().parent, *Path(__file__).resolve().parents, Path.cwd()]
             if (p/'data/processed/training_manifest.json').exists()), Path.cwd())
DATA = ROOT / 'models/claude_image_price_tuning_20261022/train_val'
OLD = ROOT / 'models/image_price_20261022T114653955552'
SOURCE = ROOT / 'tmp/independent_review_20261023/gemini/experiment_v2_20261022'
POINTER = ROOT / 'models/image_price_final_work.json'
SEED = 20261023
torch.set_num_threads(4)
threadpool_limits(4)


def digest(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(1048576), b''):
            h.update(b)
    return h.hexdigest()


def js(p, obj=None):
    if obj is None:
        return json.loads(Path(p).read_text(encoding='utf-8-sig'))
    Path(p).write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def rows(p):
    with Path(p).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def csvwrite(p, rs):
    with Path(p).open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rs[0]))
        w.writeheader()
        w.writerows(rs)


def modules(run):
    sys.path.insert(0, str(run / 'vendor'))
    import gemini_features as gf
    import gemini_training as gt
    return gf, gt


def check(run=None):
    from improve_common import compare_inputs
    audit, _ = compare_inputs()
    if audit['critical'] or not audit['split_matches_manifest']:
        raise ValueError('اختلاف التقسيم أو group_id')
    manifest = js(ROOT / 'data/processed/training_manifest.json')
    for name, sha in manifest['input_files_sha256'].items():
        if digest(ROOT / 'data/processed' / name) != sha:
            raise ValueError('تغيرت جداول المصدر: ' + name)
    if run:
        for rel, sha in js(run / 'input_identity.json')['files'].items():
            if digest(ROOT / rel) != sha:
                raise ValueError('تغير مدخل معتمد: ' + rel)
    return audit


def prepare():
    audit = check()
    run = ROOT / 'models' / ('image_price_final_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S'))
    (run / 'vendor').mkdir(parents=True)
    (run / 'cache').mkdir()
    shutil.copy2(SOURCE / 'image_price_runtime.py', run / 'vendor/gemini_features.py')
    # Preserve the upstream training helper unchanged; import expects its original runtime name.
    shutil.copy2(SOURCE / 'image_price_runtime.py', run / 'vendor/image_price_runtime.py')
    shutil.copy2(SOURCE / 'image_price_train_v2.py', run / 'vendor/gemini_training.py')
    shutil.copy2(OLD / 'encoder_weights.pth', run / 'encoder_weights.pth')
    (run / 'gemini').mkdir()
    for name in ['runtime_policy.json', 'v2_model_artifacts.npz', 'domain.joblib', 'val_predictions.csv', 'LOCK.json']:
        shutil.copy2(SOURCE / name, run / 'gemini' / name)
    d = rows(DATA / 'dataset.csv')
    assert Counter(r['split'] for r in d) == {'train': 1429, 'val': 321}
    assert len({r['sample_id'] for r in d}) == len(d)
    train = {r['group_id'] for r in d if r['split'] == 'train'}
    val = {r['group_id'] for r in d if r['split'] == 'val'}
    assert not train & val
    seen = {}
    for r in d:
        assert digest(DATA / r['image_path']) == r['sha256']
        if r['sha256'] in seen and seen[r['sha256']] != r['split']:
            raise ValueError('SHA يعبر الأقسام')
        seen[r['sha256']] = r['split']
    freeze = [DATA/'dataset.csv', OLD/'encoder_weights.pth', OLD/'price_network.pt',
              ROOT/'الملفات_المهمة/الداتا_المحدثة.csv', ROOT/'الملفات_المهمة/ربط_الصور.csv',
              ROOT/'data/processed/listings.csv', ROOT/'data/processed/images.csv',
              ROOT/'data/processed/training_manifest.json']
    js(run/'input_identity.json', {'files': {p.relative_to(ROOT).as_posix():digest(p) for p in freeze},
        'audit':audit, 'split_fingerprint':js(ROOT/'data/processed/training_manifest.json')['split_fingerprint'],
        'image_hashes_verified':len(d), 'test_accessed':False})
    js(run/'protocol.json', {'selection':'val MAE; MAPE <= Gemini+0.10pp; hit20 >= Gemini-1pp; low/mid band safeguards',
        'maximum_new_head_trials':16, 'maximum_blend_trials':4, 'test_policy':'one final selected model only, after lock',
        'historical_test_exposed':True, 'features':'single image pixels only; group_id not a feature',
        'no_new_images_or_labels':True, 'seed':SEED, 'source':'Gemini V2 helpers preserved in vendor',
        'source_sha256':{p.name:digest(p) for p in (run/'vendor').glob('*.py')}})
    js(POINTER, {'run_dir':run.relative_to(ROOT).as_posix()})
    print(run, flush=True)


def extract(run):
    check(run)
    if (run/'test_started.json').exists():
        raise ValueError('انتهى اختيار النماذج')
    gf, _ = modules(run)
    import cv2
    if 'site-packages' not in str(cv2.__file__):
        raise ValueError('يجب استخدام OpenCV الفعلي')
    d = rows(DATA/'dataset.csv'); tr = np.array([r['split']=='train' for r in d])
    art = np.load(run/'gemini/v2_model_artifacts.npz')
    vs = np.stack([gf.preprocess_image_views(gf.load_image(DATA/r['image_path']))[2] for r in d])
    mean = vs[tr].astype('float64').mean(0,keepdims=True).astype('float32')
    std = np.maximum(vs[tr].astype('float64').std(0,keepdims=True),1e-6).astype('float32')
    assert np.max(np.abs(mean-art['vs_mean'])) < 1e-4
    assert np.max(np.abs(std-art['vs_std'])) < 1e-4
    arrays={key:np.lib.format.open_memmap(run/'cache'/f'{key}.npy',mode='w+',dtype='float32',shape=(len(d),dim))
            for key,dim in [('crop',25154),('full',21504),('x53',15170),('domain',512)]}
    back=gf.BackboneRunner(run)
    t=time.perf_counter()
    for start in range(0,len(d),16):
        batch=d[start:start+16]
        prep=[gf.preprocess_image_views(gf.load_image(DATA/r['image_path'])) for r in batch]
        xc=np.stack([p[0] for p in prep]); xf=np.stack([p[1] for p in prep])
        outs=[back.run_view(v) for v in [xc,xc[:,:,:,::-1].copy(),xf,xf[:,:,:,::-1].copy()]]
        a,b,c,e=outs
        fdom,crop,full,x53=gf.build_descriptors_from_backbone_outputs(a[0],a[1],a[2],a[3],a[4],
            b[2],b[3],b[4],c[2],c[3],c[4],e[2],e[3],e[4],vs[start:start+len(batch)],mean,std)
        for key,val in [('crop',crop),('full',full),('x53',x53),('domain',fdom)]:
            arrays[key][start:start+len(batch)]=val.astype('float32')
        if start%160==0:print(f'extracted {start+len(batch)}/{len(d)} seconds={time.perf_counter()-t:.1f}',flush=True)
    for a in arrays.values():a.flush()
    np.savez(run/'cache/stats.npz',vs=vs,vs_mean=mean,vs_std=std)
    js(run/'cache/identity.json',{'dataset_sha256':digest(DATA/'dataset.csv'),'cv2':cv2.__version__,
        'opencv_path':cv2.__file__, 'seconds':time.perf_counter()-t,
        'files':{p.name:digest(p) for p in (run/'cache').glob('*.npy')}})
    check(run)


def feats(run):
    if digest(DATA/'dataset.csv')!=js(run/'cache/identity.json')['dataset_sha256']:
        raise ValueError('اختلاف هوية المخزن')
    return {key:np.load(run/'cache'/f'{key}.npy',mmap_mode='r') for key in ['crop','full','x53','domain']}


def met(y,p):
    e=np.abs(y-p); ape=e/y*100
    return {'MAE_SAR':float(e.mean()),'RMSE_SAR':float(np.sqrt(np.mean((y-p)**2))),
        'MAPE_percent':float(ape.mean()),'median_APE_percent':float(np.median(ape)),
        'R2':float(1-np.sum((y-p)**2)/np.sum((y-y.mean())**2)),
        'within_20_percent':float(np.mean(ape<=20)*100),'n':len(y)}


def adjust(a, raw=None, mean=None, std=None, nostats=False):
    x=np.array(a,dtype='float32',copy=True)
    if nostats:
        x[:,-154:]=0
    elif raw is not None:
        prefix=x[:,:-154]
        prefix*=np.sqrt(1-.1/1.95)/np.maximum(np.linalg.norm(prefix,axis=1,keepdims=True),1e-12)
        v=np.clip((raw-mean)/std,-5,5)
        v/=np.maximum(np.linalg.norm(v,axis=1,keepdims=True),1e-12)
        x[:,-154:]=v*np.sqrt(.1/1.95)
    x/=np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-12)
    return x


def similarities(F, ai,bi, nostats=False, foldstats=None):
    out=[]
    for name in ['crop','full','x53']:
        a=np.array(F[name][ai],dtype='float32'); b=np.array(F[name][bi],dtype='float32')
        if name!='full':
            if foldstats:
                raw,mean,std=foldstats
                a=adjust(a,raw[ai],mean,std,nostats);b=adjust(b,raw[bi],mean,std,nostats)
            elif nostats:
                a=adjust(a,nostats=True);b=adjust(b,nostats=True)
        out.append(np.clip((a@b.T).astype('float64'),-1,1))
    return out


def kernel(S, spec):
    crop,full,s53=S
    a,b,c=spec['components']; mix=spec['full_weight']
    return a*((1-mix)*crop+mix*full)+b*s53**2+c*np.exp(-spec['gamma']*np.maximum(2-2*s53,0))


def fit(gt,K,KV,y,w,spec):
    if spec.get('estimator')=='svr':
        m=SVR(C=spec['C'],epsilon=spec.get('epsilon',.05),kernel='precomputed')
        sample_w=w*np.clip(y/np.median(y),.25,4.)**spec.get('price_power',0.)
        sample_w/=sample_w.mean()
        m.fit(K,np.log(y),sample_weight=sample_w)
        coef=np.zeros(len(y));coef[m.support_]=m.dual_coef_[0]
        b=float(m.intercept_[0]);bounds=np.log([y.min(),y.max()])
        return np.exp(np.clip(K@coef+b,*bounds)),np.exp(np.clip(KV@coef+b,*bounds)),coef,b
    return gt.fit_rkhs(K,KV,y,w,alpha=spec['alpha'],w_log=spec['w_log'],w_sar=1-spec['w_log'])


def baseline(run,F,ii):
    art=np.load(run/'gemini/v2_model_artifacts.npz')
    x=F['x53'][ii].astype('float64')
    v=(art['Vq'].astype('float32')*art['scale_v']).astype('float64')
    z=float(art['b_tot'][0])+F['crop'][ii]@art['w_crop_7x4'].astype('float64')+F['full'][ii]@art['w_full_7x4'].astype('float64')
    z+=x@art['w_53_lin'].astype('float64')+((x@v)**2)@art['lk'].astype('float64')
    return np.exp(np.clip(z,*art['log_prediction_bounds']))


def search(run):
    check(run)
    if (run/'LOCK.json').exists():raise ValueError('النموذج مقفل؛ لا مزيد من الاختيار')
    gf,gt=modules(run);F=feats(run);d=rows(DATA/'dataset.csv')
    ti=np.array([i for i,r in enumerate(d) if r['split']=='train']);vi=np.array([i for i,r in enumerate(d) if r['split']=='val'])
    yt=np.array([float(d[i]['price_sar']) for i in ti]);yv=np.array([float(d[i]['price_sar']) for i in vi])
    counts=Counter(d[i]['group_id'] for i in ti); w=np.array([1/counts[d[i]['group_id']] for i in ti]);w/=w.mean()
    original=baseline(run,F,vi);bm=met(yv,original)
    stored={r['sample_id']:float(r['predicted_sar']) for r in rows(run/'gemini/val_predictions.csv')}
    maxdiff=float(np.max(np.abs(original-np.array([stored[d[i]['sample_id']] for i in vi]))))
    js(run/'gemini_reproduction.json',{'actual_val':bm,'max_absolute_prediction_diff_sar':maxdiff,
        'exact_opencv':True,'backbone':'PyTorch','all_321_raw_val_images':True})
    print('Gemini actual val',bm,'maxdiff',maxdiff,flush=True)
    if maxdiff>5:raise ValueError('لم تتطابق إعادة إنتاج جيمناي')
    # Finite experiment budget, declared before fitting new heads.
    specs=[]
    def add(name,alpha=.18,wl=.7,mix=.28,components=(1.,0.,0.),gamma=1.5,nostats=False,**kw):
        specs.append(dict(name=name,alpha=alpha,w_log=wl,full_weight=mix,components=list(components),gamma=gamma,nostats=nostats,**kw))
    add('linear_gemini')
    add('kernel_gemini',.22,.75,components=(.55,.25,.20))
    add('linear_lower_reg',.08)
    add('linear_higher_reg',.32)
    add('linear_balanced_loss',.18,.5)
    add('linear_more_full',.18,.7,.5)
    add('kernel_lower_reg',.10,.75,components=(.55,.25,.20))
    add('kernel_higher_reg',.40,.75,components=(.55,.25,.20))
    add('kernel_balanced_loss',.22,.5,components=(.55,.25,.20))
    add('kernel_less_nonlinear',.18,.7,components=(.8,.1,.1))
    add('linear_no_stats',nostats=True)
    add('kernel_no_stats',.22,.75,components=(.55,.25,.2),nostats=True)
    add('svr_mix_C10',components=(.55,.25,.2),estimator='svr',C=10.)
    add('svr_mix_C100',components=(.55,.25,.2),estimator='svr',C=100.)
    js(run/'search_plan.json',{'new_head_trials':specs,'blend_weights':[.25,.5],
        'maximum_blend_base_models':2,'selection_before_any_test':True})
    t=time.perf_counter(); pred={}; summary=[]
    bands=np.quantile(yt,[1/3,2/3]); binid=np.digitize(yv,bands)
    def eligible(p):
        m=met(yv,p)
        ok=m['MAE_SAR']<bm['MAE_SAR'] and m['MAPE_percent']<=bm['MAPE_percent']+.1 and m['within_20_percent']>=bm['within_20_percent']-1.
        for k in [0,1]:
            ix=binid==k
            ok &= np.mean(np.abs(yv[ix]-p[ix])) <= np.mean(np.abs(yv[ix]-original[ix]))*1.15
            ok &= np.mean(np.abs(yv[ix]-p[ix])/yv[ix])*100 <= np.mean(np.abs(yv[ix]-original[ix])/yv[ix])*100+2.
        return bool(ok)
    for nostats in [False,True]:
        ST=similarities(F,ti,ti,nostats); SV=similarities(F,vi,ti,nostats)
        for spec in [s for s in specs if s['nostats']==nostats]:
            tt=time.perf_counter();k=kernel(ST,spec);kv=kernel(SV,spec)
            ptr,pva,c,b=fit(gt,k,kv,yt,w,spec)
            np.savez(run/'cache'/f"head_{spec['name']}.npz",coef=c,intercept=[b],train=ptr,val=pva)
            m=met(yv,pva);pred[spec['name']]=pva
            summary.append({'candidate':spec['name'],'kind':'head',**m,'train_MAE_SAR':met(yt,ptr)['MAE_SAR'],
                'eligible':eligible(pva),'seconds':time.perf_counter()-tt})
            csvwrite(run/'experiments.csv',summary)
            print(spec['name'],round(m['MAE_SAR'],2),round(m['MAPE_percent'],2),'eligible',eligible(pva),flush=True)
    tops=sorted(summary,key=lambda x:x['MAE_SAR'])[:2]
    blends=[]
    for s in tops:
        for weight in [.25,.5]:
            p=np.exp((1-weight)*np.log(pred[s['candidate']])+weight*np.log(original))
            name=f"blend_{s['candidate']}_{weight}"
            spec={'name':name,'head':s['candidate'],'gemini_weight':weight}
            blends.append(spec);pred[name]=p
            summary.append({'candidate':name,'kind':'blend',**met(yv,p),'train_MAE_SAR':float('nan'),
                'eligible':eligible(p),'seconds':0.})
    csvwrite(run/'experiments.csv',summary)
    candidates=[s for s in summary if s['eligible']]
    winner=min(candidates,key=lambda s:s['MAE_SAR']) if candidates else {'candidate':'original_gemini','kind':'gemini',**bm}
    choice=winner['candidate']; spec=next((s for s in specs if s['name']==choice),None)
    blend=next((s for s in blends if s['name']==choice),None)
    if blend:spec=next(s for s in specs if s['name']==blend['head'])
    selected={'choice':choice,'val_metrics':met(yv,pred[choice] if choice!='original_gemini' else original),
        'head_spec':spec,'blend':blend,'gemini_reference':bm,'price_band_bounds_train':bands.tolist(),
        'elapsed_seconds':time.perf_counter()-t,'new_trial_count':len(summary),'test_not_evaluated':True}
    js(run/'selection.json',selected)
    p=pred[choice] if choice!='original_gemini' else original
    csvwrite(run/'val_predictions.csv',[{'sample_id':d[i]['sample_id'],'group_id':d[i]['group_id'],
        'actual_sar':float(y),'predicted_sar':float(v),'gemini_sar':float(g)} for i,y,v,g in zip(vi,yv,p,original)])
    print('SELECTED',json.dumps(selected),flush=True)
    check(run)


def crossvalidate(run):
    check(run);sel=js(run/'selection.json');spec=sel['head_spec']
    if spec is None:
        js(run/'crossvalidation.json',{'status':'kept_original_gemini','no_new_head':True});return
    _,gt=modules(run);F=feats(run);d=rows(DATA/'dataset.csv')
    ti=np.array([i for i,r in enumerate(d) if r['split']=='train']);tr=[d[i] for i in ti]
    yt=np.array([float(r['price_sar']) for r in tr]);oof=np.zeros(len(tr))
    raw=np.load(run/'cache/stats.npz')['vs'].astype('float64')
    for n,(fit_ix,hold_ix) in enumerate(gt.get_group_kfold(tr,3)):
        ai,bi=ti[fit_ix],ti[hold_ix]
        mean=raw[ai].mean(0,keepdims=True);std=np.maximum(raw[ai].std(0,keepdims=True),1e-6)
        st=similarities(F,ai,ai,spec['nostats'],(raw,mean,std))
        sh=similarities(F,bi,ai,spec['nostats'],(raw,mean,std))
        counts=Counter(d[i]['group_id'] for i in ai);w=np.array([1/counts[d[i]['group_id']] for i in ai]);w/=w.mean()
        _,p,_,_=fit(gt,kernel(st,spec),kernel(sh,spec),yt[fit_ix],w,spec);oof[hold_ix]=p
        print('CV fold',n+1,met(yt[hold_ix],p),flush=True)
    np.save(run/'cache/selected_oof.npy',oof)
    js(run/'crossvalidation.json',{'out_of_fold':met(yt,oof),'folds':3,
        'group_disjoint':True,'visual_stats_fitted_inside_each_fold':True,
        'candidate_selected_on_val':True,'applies_to_head_only_not_gemini_blend':bool(sel['blend']),
        'target_bounds_fit_on_fold_train_only':True})


def refine_cv(run):
    """آخر مرشحين: اختيار C/epsilon/أوزان الخسارة على CV التدريب وحده."""
    check(run)
    if (run/'LOCK.json').exists() or (run/'refine_cv_completed.json').exists():
        raise ValueError('لا تكرر البحث أو تفتح الاختيار بعد القفل')
    _,gt=modules(run);F=feats(run);d=rows(DATA/'dataset.csv')
    ti=np.array([i for i,r in enumerate(d) if r['split']=='train']);vi=np.array([i for i,r in enumerate(d) if r['split']=='val'])
    yt=np.array([float(d[i]['price_sar']) for i in ti]);yv=np.array([float(d[i]['price_sar']) for i in vi])
    tr=[d[i] for i in ti];raw=np.load(run/'cache/stats.npz')['vs'].astype('float64')
    base=dict(alpha=.18,w_log=.7,full_weight=.28,components=[.55,.25,.20],gamma=1.5,nostats=False,estimator='svr')
    grid=[dict(base,C=C,epsilon=e,price_power=p,name=f'cv_svr_C{C}_eps{e}_pw{p}')
          for p in [0.,.5] for C in [1.,10.,100.] for e in [.05,.12]]
    js(run/'refine_cv_plan.json',{'grid':grid,'selection':'lowest train GroupKFold OOF MAE per loss-weight family; only two val evaluations',
        'number_of_new_val_candidates':2,'no_test_used':True})
    out={s['name']:np.zeros(len(ti)) for s in grid}
    for n,(fit_ix,hold_ix) in enumerate(gt.get_group_kfold(tr,3)):
        ai,bi=ti[fit_ix],ti[hold_ix];mean=raw[ai].mean(0,keepdims=True);std=np.maximum(raw[ai].std(0,keepdims=True),1e-6)
        K=kernel(similarities(F,ai,ai,False,(raw,mean,std)),base)
        KH=kernel(similarities(F,bi,ai,False,(raw,mean,std)),base)
        counts=Counter(d[i]['group_id'] for i in ai);w=np.array([1/counts[d[i]['group_id']] for i in ai]);w/=w.mean()
        for spec in grid:
            _,p,_,_=fit(gt,K,KH,yt[fit_ix],w,spec);out[spec['name']][hold_ix]=p
        print('refine train-CV fold',n+1,'done',flush=True)
    scores=[{'candidate':s['name'],'price_power':s['price_power'],**met(yt,out[s['name']])} for s in grid]
    csvwrite(run/'refine_train_cv.csv',scores)
    chosen=[]
    for power in [0.,.5]:
        best=min([s for s in scores if s['price_power']==power],key=lambda s:s['MAE_SAR'])
        chosen.append(next(s for s in grid if s['name']==best['candidate']))
    K=kernel(similarities(F,ti,ti),base);KV=kernel(similarities(F,vi,ti),base)
    counts=Counter(d[i]['group_id'] for i in ti);w=np.array([1/counts[d[i]['group_id']] for i in ti]);w/=w.mean()
    gem=baseline(run,F,vi);bm=met(yv,gem);sel=js(run/'selection.json');experiments=rows(run/'experiments.csv')
    bands=np.array(sel['price_band_bounds_train']);binid=np.digitize(yv,bands)
    for spec in chosen:
        tt=time.perf_counter();ptr,pva,c,b=fit(gt,K,KV,yt,w,spec);m=met(yv,pva)
        ok=m['MAE_SAR']<bm['MAE_SAR'] and m['MAPE_percent']<=bm['MAPE_percent']+.1 and m['within_20_percent']>=bm['within_20_percent']-1.
        for k in [0,1]:
            ix=binid==k
            ok &= np.mean(np.abs(yv[ix]-pva[ix])) <= np.mean(np.abs(yv[ix]-gem[ix]))*1.15
            ok &= np.mean(np.abs(yv[ix]-pva[ix])/yv[ix])*100 <= np.mean(np.abs(yv[ix]-gem[ix])/yv[ix])*100+2.
        np.savez(run/'cache'/f"head_{spec['name']}.npz",coef=c,intercept=[b],train=ptr,val=pva)
        experiments.append({'candidate':spec['name'],'kind':'head_cv_selected',**m,'train_MAE_SAR':met(yt,ptr)['MAE_SAR'],
            'eligible':bool(ok),'seconds':time.perf_counter()-tt})
        print('CV-selected val',spec['name'],m,'eligible',ok,flush=True)
        if ok and m['MAE_SAR']<sel['val_metrics']['MAE_SAR']:
            sel.update(choice=spec['name'],val_metrics=m,head_spec=spec,blend=None)
            csvwrite(run/'val_predictions.csv',[{'sample_id':d[i]['sample_id'],'group_id':d[i]['group_id'],
                'actual_sar':float(y),'predicted_sar':float(v),'gemini_sar':float(g)} for i,y,v,g in zip(vi,yv,pva,gem)])
    csvwrite(run/'experiments.csv',experiments);sel['new_trial_count']=len(experiments);js(run/'selection.json',sel)
    js(run/'refine_cv_completed.json',{'selected_candidates_on_train_cv':chosen,'final_selection':sel['choice']})


def export(run):
    check(run)
    sel=js(run/'selection.json');spec=sel['head_spec']
    if spec:
        d=rows(DATA/'dataset.csv');ti=np.array([i for i,r in enumerate(d) if r['split']=='train'])
        F=feats(run);name=spec['name'];h=np.load(run/'cache'/f'head_{name}.npz');c=h['coef'];b=h['intercept']
        crop=np.array(F['crop'][ti]);full=np.array(F['full'][ti]);x53=np.array(F['x53'][ti])
        if spec['nostats']:crop=adjust(crop,nostats=True);x53=adjust(x53,nostats=True)
        a,p,r=spec['components'];mix=spec['full_weight']
        np.savez_compressed(run/'head.npz',w_crop=(a*(1-mix)*(crop.T.astype('float64')@c)).astype('float32'),
            w_full=(a*mix*(full.T.astype('float64')@c)).astype('float32'),intercept=b,
            support53=x53 if (p or r) else np.zeros((0,0),dtype='float32'),
            poly_coef=p*c,rbf_coef=r*c)
    shutil.copy2(ROOT/'scripts/image_price_final_runtime.py',run/'runtime.py')
    shutil.copy2(ROOT/'scripts/image_price_final.py',run/'training.py')
    stats=np.load(run/'cache/stats.npz')
    np.savez(run/'visual_stats.npz',vs_mean=stats['vs_mean'],vs_std=stats['vs_std'])
    y=np.array([float(r['price_sar']) for r in rows(DATA/'dataset.csv') if r['split']=='train'])
    js(run/'policy.json',{'status':'experimental','spec':spec,'blend':sel['blend'],'bounds':np.log([y.min(),y.max()]).tolist(),
        'pixels_only':True,'listing_database_required':False,'uncalibrated_domain_check':True})
    print('exported',run,flush=True)


def verify(run):
    check(run)
    import importlib.util
    s=importlib.util.spec_from_file_location('final_runtime',run/'runtime.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
    rt=m.ImagePriceRuntime(run);d=rows(DATA/'dataset.csv');v=[r for r in d if r['split']=='val']
    stored=rows(run/'val_predictions.csv');expected={r['sample_id']:float(r['predicted_sar']) for r in stored}
    dif=[];relative=[];actual=[];tt=time.perf_counter()
    for start in range(0,len(v),8):
        rr=v[start:start+8];p,_=rt.predict_batch_paths([DATA/r['image_path'] for r in rr])
        for r,a in zip(rr,p):
            delta=abs(float(a)-expected[r['sample_id']]);dif.append(delta)
            relative.append(delta/max(expected[r['sample_id']],1.));actual.append(float(a))
    # Kernel coefficients have cancellation in float32 Gram matrices versus the
    # exported primal weights. Verify bounded rounding differences, then report
    # actual release-runtime predictions for every val sample before locking.
    result={'raw_val_images':len(v),'max_price_diff_sar':max(dif),'max_relative_difference':max(relative),
        'tolerance_sar':10.,'tolerance_relative':.0001,'seconds':time.perf_counter()-tt,
        'passed':max(dif)<10. and max(relative)<.0001}
    js(run/'runtime_verification.json',result)
    if not result['passed']:raise ValueError('عدم تطابق مشغل التسليم')
    for row,p in zip(stored,actual):row['predicted_sar']=p
    csvwrite(run/'val_predictions.csv',stored)
    sel=js(run/'selection.json');sel.setdefault('val_metrics_before_export',sel['val_metrics'])
    sel['val_metrics']=met(np.array([float(r['actual_sar']) for r in stored]),np.array(actual));js(run/'selection.json',sel)
    try:rt.predict(ROOT/'REQ.md');raise AssertionError('accepted invalid image')
    except ValueError:pass
    print('runtime',result,flush=True)


def lock(run):
    check(run)
    if not js(run/'runtime_verification.json')['passed']:raise ValueError('استدلال غير ناجح')
    names=['policy.json','runtime.py','training.py','selection.json','visual_stats.npz','encoder_weights.pth',
           'vendor/gemini_features.py','vendor/gemini_training.py','vendor/image_price_runtime.py',
           'gemini/v2_model_artifacts.npz','gemini/runtime_policy.json','gemini/domain.joblib']
    if (run/'head.npz').exists():names.append('head.npz')
    with (run/'LOCK.json').open('x',encoding='utf-8') as f:
        json.dump({'at':datetime.now(timezone.utc).isoformat(),'test_not_evaluated':True,
            'selection':js(run/'selection.json'),'sha256':{n:digest(run/n) for n in names}},f,ensure_ascii=False,indent=2)
    print('LOCKED',flush=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('command',choices=['prepare','extract','search','refine-cv','cv','export','verify','lock'])
    ap.add_argument('--run',type=Path);args=ap.parse_args()
    if args.command=='prepare':return prepare()
    run=args.run or ROOT/js(POINTER)['run_dir']
    {'extract':extract,'search':search,'refine-cv':refine_cv,'cv':crossvalidate,'export':export,'verify':verify,'lock':lock}[args.command](run)


if __name__=='__main__':main()
