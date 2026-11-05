"""Foreground DINO patch pooling + masked color features; local experiments only."""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='2'
import argparse,json,time
from datetime import datetime,timezone
from pathlib import Path
from collections import Counter
import numpy as np
import joblib
import torch
from torch.nn import functional as F
from PIL import Image,ImageOps,ImageDraw
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torchvision.models.segmentation import lraspp_mobilenet_v3_large,LRASPP_MobileNet_V3_Large_Weights
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier
from threadpoolctl import threadpool_limits
import robust_visual_round_20261026 as r

OUT=r.ROOT/'models/foreground_visual_20261026_v1'
torch.set_num_threads(4)

def setup():
    OUT.mkdir(exist_ok=True)
    if not (OUT/'protocol.json').exists():
        r.save(OUT/'protocol.json',{'seed':r.SEED,'segmentation':'Frozen torchvision LRASPP MobileNetV3 COCO/VOC; car category; RGB resize256 square, ImageNet norm',
            'color':'HSV/Lab quantiles and histograms within foreground + spatial body bands; bands are heuristics, not paint segmentation',
            'patch':'Frozen DINOv2 ViT-S/14, same resize256/center224; car-mask weighted patch mean and std',
            'fallback':'car mask <3% pixels: existing detector bounding rectangle, then whole frame if unavailable',
            'CV':'same three train group folds; equal listing-weighted macro-F1, one selected candidate per task',
            'train_gate':'macro-F1 +.01 and accuracy decline <=.01 vs current direct DINO component',
            'val_gate':'historical val, macro-F1 +.01 and positive paired group bootstrap CI, accuracy decline <=.01; development only',
            'inference':'single image','test_used':False,'external12_for_selection':False,
            'data_identity':r.v.load_json(r.ROOT/'data/processed/training_manifest.json')})

def models():
    weights=LRASPP_MobileNet_V3_Large_Weights.DEFAULT
    state=torch.hub.load_state_dict_from_url(weights.url,model_dir=str(OUT/'weights'),check_hash=True,progress=False)
    seg=lraspp_mobilenet_v3_large(weights=None,weights_backbone=None).eval().requires_grad_(False)
    seg.load_state_dict(state)
    d=torch.hub.load(str(r.v.HUB/'facebookresearch_dinov2_main'),'dinov2_vits14',source='local',pretrained=False)
    d.load_state_dict(torch.load(r.v.HUB/'checkpoints/dinov2_vits14_pretrain.pth',map_location='cpu',weights_only=True))
    d.eval().requires_grad_(False)
    tf=transforms.Compose([transforms.Resize(256,interpolation=InterpolationMode.BICUBIC),transforms.CenterCrop(224),
        transforms.ToTensor(),transforms.Normalize((.485,.456,.406),(.229,.224,.225))])
    mtf=transforms.Compose([transforms.Resize(256,interpolation=InterpolationMode.BILINEAR),transforms.CenterCrop(224)])
    return seg,d,tf,mtf

def paint(rgb,mask,quality):
    import cv2
    cv2.setNumThreads(1)
    hsv=cv2.cvtColor(rgb,cv2.COLOR_RGB2HSV)/np.array([179.,255.,255.])
    lab=cv2.cvtColor(rgb,cv2.COLOR_RGB2LAB)/255.
    z=np.concatenate([hsv,lab],axis=2)
    yy,xx=np.where(mask);y0,y1=yy.min(),yy.max()+1;x0,x1=xx.min(),xx.max()+1
    gy,gx=np.indices(mask.shape);ry=(gy-y0)/max(y1-y0,1);rx=(gx-x0)/max(x1-x0,1)
    regions=[mask,mask&(ry>=.35)&(ry<=.85)&(rx>=.05)&(rx<=.95),mask&(ry>=.45)&(ry<=.7)]
    out=[]
    for region in regions:
        if region.sum()<32:region=mask
        a=z[region]
        out.extend(np.quantile(a,[.1,.25,.5,.75,.9],axis=0).ravel())
        for j in range(6):out.extend(np.histogram(a[:,j],bins=12,range=(0,1))[0]/len(a))
        out.append(float(region.mean()))
    out.extend(quality)
    return np.asarray(out,dtype='float32')

def batch_features(rows,mods,boxes):
    seg,d,tf,mtf=mods;images=[];rgb=[];inputs=[]
    for row in rows:
        with Image.open(r.ROOT/'الملفات_المهمة'/row['relative_path']) as src:im=ImageOps.exif_transpose(src).convert('RGB')
        images.append(im);small=np.asarray(im.resize((256,256),Image.Resampling.BILINEAR)).copy();rgb.append(small)
        t=torch.from_numpy(small).permute(2,0,1).float()/255.
        inputs.append((t-t.new_tensor([.485,.456,.406])[:,None,None])/t.new_tensor([.229,.224,.225])[:,None,None])
    with torch.inference_mode():
        logits=seg(torch.stack(inputs))['out'];prob=logits.softmax(1)[:,7].numpy();labels=logits.argmax(1).numpy()
        feats=d.forward_features(torch.stack([tf(im) for im in images]))
    patch=feats['x_norm_patchtokens'].numpy();pool=[];colors=[];stats=[];masks=[]
    for i,(im,row) in enumerate(zip(images,rows)):
        mask=labels[i]==7;fraction=float(mask.mean());fallback=fraction<.03
        if fallback:
            mask=np.zeros((256,256),dtype=bool);b=boxes.get(row['sha256'])
            if b and str(b['found'])=='True':
                x1,y1,x2,y2=json.loads(b['box']);w,h=im.size
                mask[max(0,int(y1/h*256)):min(256,int(y2/h*256)+1),max(0,int(x1/w*256)):min(256,int(x2/w*256)+1)]=True
            if mask.sum()<32:mask[:]=True
            weighted=mask.astype('float32')
        else:weighted=prob[i]*mask
        colors.append(paint(rgb[i],mask,[fraction,float(fallback)]))
        m=Image.fromarray(weighted.astype('float32')).resize(im.size,Image.Resampling.BILINEAR)
        m=np.asarray(mtf(m)).copy();pw=F.interpolate(torch.from_numpy(m)[None,None],size=(16,16),mode='area').numpy().reshape(-1)
        pw=pw/max(pw.sum(),1e-8)
        mean=(patch[i]*pw[:,None]).sum(0);std=np.sqrt(np.maximum(((patch[i]-mean)**2*pw[:,None]).sum(0),0))
        pool.append(np.r_[mean,std]);masks.append(mask)
        stats.append({'image_uid':row['image_uid'],'sha256':row['sha256'],'car_fraction':fraction,'fallback':fallback})
    return np.stack(pool),np.stack(colors),stats,rgb,masks,feats['x_norm_clstoken'].numpy()

def rows_for(split,tasks=None):
    return sorted({z['sha256']:z for t in (tasks or r.TASKS) for z in r.v.compact_rows(t,split)}.values(),key=lambda z:z['image_uid'])

def preview():
    rows=rows_for('train');groups={}
    for row in rows:groups.setdefault(row['group_id'],row)
    group_rows=list(groups.values());ids=np.random.default_rng(r.SEED).choice(len(group_rows),12,replace=False)
    sample=[group_rows[i] for i in ids];mods=models();boxes={z['sha256']:z for z in r.v.read_csv(r.v.RUN/'features/boxes.csv')}
    start=time.perf_counter();all_stats=[];canvas=Image.new('RGB',(4*256,3*280),'white');draw=ImageDraw.Draw(canvas)
    for starti in range(0,12,4):
        pp,cc,st,rgb,masks,cls=batch_features(sample[starti:starti+4],mods,boxes);all_stats.extend(st)
        for j,(im,mask) in enumerate(zip(rgb,masks)):
            n=starti+j;over=im.copy();over[~mask]=(over[~mask]*.18).astype('uint8')
            canvas.paste(Image.fromarray(over),((n%4)*256,(n//4)*280));draw.text(((n%4)*256+3,(n//4)*280+257),sample[n]['image_uid'][-18:],fill='black')
        assert np.isfinite(pp).all() and np.isfinite(cc).all()
    canvas.save(OUT/'mask_preview_train.png');elapsed=time.perf_counter()-start
    r.save(OUT/'preview.json',{'images':12,'seconds':elapsed,'seconds_per_image':elapsed/12,'stats':all_stats})
    print('PREVIEW',elapsed,'seconds; dims',pp.shape,cc.shape,flush=True)

def extract(split,tasks=None):
    rows=rows_for(split,tasks);folder=OUT/('features_'+split);folder.mkdir(exist_ok=True)
    if (folder/'index.csv').exists():assert [z['sha256'] for z in r.v.read_csv(folder/'index.csv')]==[z['sha256'] for z in rows]
    else:r.table(folder/'index.csv',rows)
    done=r.v.load_json(folder/'progress.json')['done'] if (folder/'progress.json').exists() else 0
    if done==len(rows):return
    mods=models();boxes={z['sha256']:z for z in r.v.read_csv(r.v.RUN/'features/boxes.csv')};started=time.perf_counter()
    mode='r+' if (folder/'patch.npy').exists() else 'w+'
    pa=np.lib.format.open_memmap(folder/'patch.npy',mode=mode,dtype='float32',shape=(len(rows),768))
    co=np.lib.format.open_memmap(folder/'paint.npy',mode=mode,dtype='float32',shape=(len(rows),311))
    stats=r.v.read_csv(folder/'stats.csv') if (folder/'stats.csv').exists() else []
    for i in range(done,len(rows),8):
        batch=rows[i:i+8];pp,cc,ss,_,_,_=batch_features(batch,mods,boxes)
        assert cc.shape[1]==311 and np.isfinite(pp).all() and np.isfinite(cc).all()
        pa[i:i+len(batch)]=pp;co[i:i+len(batch)]=cc;stats.extend(ss)
        if (i-done)%128==0 or i+len(batch)==len(rows):
            pa.flush();co.flush();r.table(folder/'stats.csv',stats)
            r.save(folder/'progress.json',{'done':i+len(batch),'total':len(rows),'seconds_this_run':time.perf_counter()-started})
            print('FOREGROUND',split,i+len(batch),'/',len(rows),'seconds',round(time.perf_counter()-started,1),flush=True)
    del pa,co

def configs(task):
    if task=='color':return [{'name':'paint_fusion_C001','C':.01,'kind':'fusion'},
        {'name':'paint_fusion_C01','C':.1,'kind':'fusion'}, {'name':'paint_trees','kind':'paint','C':None}]
    return [{'name':'patch_mean_C01','C':.1,'kind':'mean'}, {'name':'patch_mean_C1','C':1.,'kind':'mean'},
        {'name':'patch_moments_C01','C':.1,'kind':'moments'}]

def features(rows,task,cfg,split):
    folder=OUT/('features_'+split);lookup={z['sha256']:i for i,z in enumerate(r.v.read_csv(folder/'index.csv'))};ix=[lookup[z['sha256']] for z in rows]
    paintf=np.load(folder/'paint.npy',mmap_mode='r')[ix];patch=np.load(folder/'patch.npy',mmap_mode='r')[ix]
    selected=r.v.load_json(r.v.RUN/task/'selection_lock.json')
    old=r.v.feats(rows,selected['kind'],task)
    if task=='color':return paintf if cfg['kind']=='paint' else np.c_[old,paintf]
    return np.c_[old,patch[:,:384] if cfg['kind']=='mean' else patch]

def single_image_features(f,patch,paint,task,art):
    """Same feature contract as the cached training path, for an original image."""
    from visual_fusion_predict import features as old_features
    selected=r.v.load_json(r.v.RUN/task/'selection_lock.json')
    old=old_features(f,selected['kind'],task);kind=art['config']['kind']
    if task=='color':return paint[None] if kind=='paint' else np.c_[old,paint[None]]
    return np.c_[old,(patch[:384] if kind=='mean' else patch)[None]]

def fit(x,rows,task,cfg):
    gc=Counter(z['group_id'] for z in rows);w=np.array([1/gc[z['group_id']] for z in rows]);w/=w.mean()
    sc=StandardScaler().fit(x);z=sc.transform(x)
    if cfg['C'] is None:
        model=ExtraTreesClassifier(n_estimators=250,max_features=.8,min_samples_leaf=5,class_weight='balanced',n_jobs=2,random_state=r.SEED)
    else:model=LogisticRegression(C=cfg['C'],class_weight='balanced',max_iter=650,random_state=r.SEED)
    model.fit(z,[z[r.TARGET[task]] for z in rows],sample_weight=w)
    return {'model':model,'scaler':sc,'classes':list(model.classes_),'config':cfg,'task':task}

def train():
    folds=r.v.load_json(r.v.RUN/'folds.json');results={};selected={};allselected={};artifacts={}
    amendment=OUT/'protocol_amendment_structure.json'
    if not amendment.exists():
        assert not (OUT/'features_val').exists() and not any((OUT/t/'val_candidate.npy').exists() for t in r.TASKS)
        policies={t:('replace_direct_component_keep_half_reference' if r.v.load_json(r.v.RUN/t/'result.json')['accepted']
            and r.v.load_json(r.v.RUN/t/'result.json')['selected_val_option']=='half_reference' else 'direct_challenger') for t in r.TASKS}
        r.save(amendment,{'created_at_utc':datetime.now(timezone.utc).isoformat(),'before_new_val_evaluation':True,
            'reason':'Preserve deployed ensemble structure and replace only its DINO direct component; no mixing coefficient selected from new val results',
            'policies':policies,'train_configs_and_gates_changed':False,'interrupted_before_val_for_scope_correction':True})
    policies=r.v.load_json(amendment)['policies']
    with threadpool_limits(limits=2):
        for task in r.TASKS:
            tr=r.v.compact_rows(task,'train');classes=sorted({z[r.TARGET[task]] for z in tr});ff=np.array([folds[z['group_id']] for z in tr])
            folder=OUT/task;folder.mkdir(exist_ok=True);scores=[]
            ref=np.load(r.OUT/task/'current_direct_oof.npy');base=r.metric(tr,ref,classes,r.TARGET[task])
            for cfg in configs(task):
                cache=folder/(cfg['name']+'_oof.npy');p=None
                if cache.exists():
                    try:
                        oldp=np.load(cache)
                        if oldp.shape==ref.shape and np.isfinite(oldp).all() and np.allclose(oldp.sum(1),1,atol=.001):p=oldp
                    except (ValueError,OSError):pass
                if p is None:
                    x=features(tr,task,cfg,'train');p=np.zeros_like(ref)
                    for fold in range(3):
                        a=np.flatnonzero(ff!=fold);b=np.flatnonzero(ff==fold);aa=[tr[i] for i in a]
                        assert not {tr[i]['group_id'] for i in a}&{tr[i]['group_id'] for i in b}
                        art=fit(x[a],aa,task,cfg);q=art['model'].predict_proba(art['scaler'].transform(x[b]));ii=[classes.index(c) for c in art['classes']]
                        p[np.ix_(b,ii)]=q
                m=r.metric(tr,p,classes,r.TARGET[task]);scores.append({'name':cfg['name'],**m});np.save(folder/(cfg['name']+'_oof.npy'),p)
                print('MASK_CV',task,cfg['name'],m,flush=True)
            best=max(scores,key=lambda z:z['macro_f1']);gate=best['macro_f1']-base['macro_f1']>=.01 and best['accuracy']>=base['accuracy']-.01
            results[task]={'reference':base,'candidates':scores,'selected':best,'train_gate':bool(gate),'val_evaluated':False,'accepted':False,'test_used':False}
            r.save(folder/'selection_lock.json',results[task])
            cfgbest=next(z for z in configs(task) if z['name']==best['name'])
            allselected[task]=cfgbest
            if gate:selected[task]=cfgbest
        r.save(OUT/'train_summary.json',results)
        # Freeze one train-selected challenger per task even when it fails the
        # promotion gate. A subsequent external-12 diagnostic cannot change it.
        for task,cfg in allselected.items():
            tr=r.v.compact_rows(task,'train');art=fit(features(tr,task,cfg,'train'),tr,task,cfg)
            art['runtime_blend_policy']=policies[task]
            artifacts[task]=art;joblib.dump(art,OUT/task/'head_train_selected.joblib',compress=3)
        r.save(OUT/'external_diagnostic_selection_lock.json',{'selection':'train CV only, before any external-12 prediction',
            'configs':allselected,'artifact_sha256':{t:r.digest(OUT/t/'head_train_selected.joblib') for t in allselected},
            'external_results_may_change_selection':False,'purpose':'qualitative diagnostic, not a performance estimate'})
        if selected:extract('val',list(selected))
        for task,cfg in selected.items():
            tr=r.v.compact_rows(task,'train');va=r.v.compact_rows(task,'val');classes=sorted({z[r.TARGET[task]] for z in tr});folder=OUT/task
            art=artifacts[task];joblib.dump(art,folder/'head.joblib',compress=3)
            raw=art['model'].predict_proba(art['scaler'].transform(features(va,task,cfg,'val')))
            new=.5*raw+.5*np.load(r.v.RUN/task/'val_reference.npy') if policies[task]=='replace_direct_component_keep_half_reference' else raw
            oldresult=r.v.load_json(r.v.RUN/task/'result.json');old=np.load(r.v.RUN/task/('val_selected.npy' if oldresult['accepted'] else 'val_reference.npy'))
            oldrows=r.v.read_csv(r.v.RUN/task/'val_predictions.csv');assert [z['image_uid'] for z in oldrows]==[z['image_uid'] for z in va]
            before=r.metric(va,old,classes,r.TARGET[task]);after=r.metric(va,new,classes,r.TARGET[task]);ci=r.paired_ci(va,old,new,classes,r.TARGET[task])
            results[task].update({'val_evaluated':True,'val_reference':before,'val_candidate':after,'paired':ci,'runtime_blend_policy':policies[task],
                'accepted':bool(after['macro_f1']-before['macro_f1']>=.01 and ci['delta_macro_f1_ci95'][0]>0 and after['accuracy']>=before['accuracy']-.01)})
            np.save(folder/'val_candidate.npy',new)
            r.table(folder/'val_predictions.csv',[{'image_uid':z['image_uid'],'group_id':z['group_id'],'actual':z[r.TARGET[task]],
                'reference':classes[old[i].argmax()],'candidate':classes[new[i].argmax()]} for i,z in enumerate(va)])
        for task,result in results.items():r.save(OUT/task/'result.json',result)
        r.save(OUT/'summary.json',results);print('FOREGROUND_RESULTS',results,flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['preview','extract','train']);args=ap.parse_args();setup()
    if args.action=='preview':preview()
    elif args.action=='extract':extract('train')
    else:train()
