"""Aspect-preserving foreground features and correct alpha decoding.

Designed after inspecting input geometry/alpha in the known diagnostic images;
those images are not an independent test. Model choices use train group CV only.
"""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='2'
import argparse,json,time
from collections import Counter
import numpy as np
import joblib
import torch
from PIL import Image,ImageOps
from torchvision import transforms
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits
import robust_visual_round_20261026 as r

OUT=r.ROOT/'models/geometry_visual_20261026_v1'
TASKS=['make','model','body']
torch.set_num_threads(4)

def decode_rgb(src):
    im=ImageOps.exif_transpose(src)
    if 'A' in im.getbands() or 'transparency' in im.info:
        rgba=im.convert('RGBA')
        return Image.alpha_composite(Image.new('RGBA',rgba.size,(255,255,255,255)),rgba).convert('RGB')
    return im.convert('RGB')

def letterbox(im):
    fitted=ImageOps.contain(im,(224,224),method=Image.Resampling.BICUBIC)
    canvas=Image.new('RGB',(224,224),'white');canvas.paste(fitted,((224-fitted.width)//2,(224-fitted.height)//2))
    return canvas

def preprocessing_checks():
    im=Image.new('RGBA',(400,100),(0,0,0,0));im.putpixel((100,50),(250,0,0,255))
    rgb=decode_rgb(im);assert rgb.getpixel((0,0))==(255,255,255) and rgb.getpixel((100,50))==(250,0,0)
    opaque=Image.new('RGB',(400,100),(200,10,20));boxed=letterbox(opaque);a=np.array(boxed)
    body=np.any(a!=255,axis=2);yy,xx=np.where(body)
    assert (xx.min(),xx.max(),yy.min(),yy.max())==(0,223,84,139)
    assert boxed.size==(224,224) and rgb.mode=='RGB'
    return {'alpha_composites_white':True,'opaque_pixels_preserved':True,'whole_image_preserved_without_center_crop':True}

def setup():
    OUT.mkdir(exist_ok=True)
    if not (OUT/'protocol.json').exists():
        r.save(OUT/'protocol.json',{'seed':r.SEED,'task_scope':TASKS,'preprocessing':'EXIF, composite transparency on white, existing car detection crop, fit entire crop into 224-square with white padding',
            'backbone':'frozen pretrained DINOv2-small CLS384; no new image data',
            'candidate_configs':['letterbox-only C .1/1','current direct representation plus letterbox C .1/1'],
            'selection':'existing train group folds, equal-listing macro-F1; +.01 gain and accuracy loss <=.01 to reach val',
            'val':'one train-selected candidate per task, preserve existing half-reference mixing structure; historical exploratory val',
            'acceptance':'+.01 val macro-F1, positive paired 95% group bootstrap CI, accuracy loss<=.01; development candidate only',
            'external12':'Known images informed diagnosis of alpha/geometry. No image-level outputs used to select C or mixing weight; not an independent test',
            'test_used':False,'data_identity':r.v.load_json(r.ROOT/'data/processed/training_manifest.json')})
    r.save(OUT/'preprocessing_checks.json',preprocessing_checks())

def load_model():
    d=torch.hub.load(str(r.v.HUB/'facebookresearch_dinov2_main'),'dinov2_vits14',source='local',pretrained=False)
    d.load_state_dict(torch.load(r.v.HUB/'checkpoints/dinov2_vits14_pretrain.pth',map_location='cpu',weights_only=True));d.eval().requires_grad_(False)
    return d,transforms.Compose([transforms.ToTensor(),transforms.Normalize((.485,.456,.406),(.229,.224,.225))])

def crop_cached(im,box):
    if box and str(box['found'])=='True':
        b=json.loads(box['box']);return im.crop(b)
    return im

def extract(split,tasks=TASKS):
    rows=sorted({z['sha256']:z for task in tasks for z in r.v.compact_rows(task,split)}.values(),key=lambda z:z['image_uid'])
    folder=OUT/('features_'+split);folder.mkdir(exist_ok=True)
    if (folder/'index.csv').exists():assert [z['sha256'] for z in r.v.read_csv(folder/'index.csv')]==[z['sha256'] for z in rows]
    else:r.table(folder/'index.csv',rows)
    done=r.v.load_json(folder/'progress.json')['done'] if (folder/'progress.json').exists() else 0
    if done==len(rows):return
    model,tf=load_model();boxes={z['sha256']:z for z in r.v.read_csv(r.v.RUN/'features/boxes.csv')}
    mode='r+' if (folder/'letterbox.npy').exists() else 'w+'
    arr=np.lib.format.open_memmap(folder/'letterbox.npy',mode=mode,dtype='float32',shape=(len(rows),384));start=time.perf_counter()
    with torch.inference_mode():
        for i in range(done,len(rows),16):
            xx=[]
            for z in rows[i:i+16]:
                with Image.open(r.ROOT/'الملفات_المهمة'/z['relative_path']) as src:im=decode_rgb(src)
                xx.append(tf(letterbox(crop_cached(im,boxes.get(z['sha256'])))))
            x=model(torch.stack(xx)).numpy();assert np.isfinite(x).all();arr[i:i+len(x)]=x
            if (i-done)%256==0 or i+len(x)==len(rows):
                arr.flush();r.save(folder/'progress.json',{'done':i+len(x),'total':len(rows),'seconds_this_run':time.perf_counter()-start})
                print('LETTERBOX',split,i+len(x),'/',len(rows),'seconds',round(time.perf_counter()-start,1),flush=True)

def configs():return [{'name':f'{kind}_C{c:g}','kind':kind,'C':c} for kind in ['letterbox','additive'] for c in [.1,1.]]

def features(rows,task,cfg,split):
    folder=OUT/('features_'+split);idx={z['sha256']:i for i,z in enumerate(r.v.read_csv(folder/'index.csv'))}
    x=np.load(folder/'letterbox.npy',mmap_mode='r')[[idx[z['sha256']] for z in rows]]
    if cfg['kind']=='letterbox':return x
    old=r.v.load_json(r.v.RUN/task/'selection_lock.json')
    return np.c_[r.v.feats(rows,old['kind'],task),x]

def fit(x,rows,task,cfg):
    cnt=Counter(z['group_id'] for z in rows);w=np.array([1/cnt[z['group_id']] for z in rows]);w/=w.mean()
    sc=StandardScaler().fit(x)
    model=LogisticRegression(C=cfg['C'],class_weight='balanced',max_iter=650,random_state=r.SEED)
    model.fit(sc.transform(x),[z[r.TARGET[task]] for z in rows],sample_weight=w)
    oldresult=r.v.load_json(r.v.RUN/task/'result.json')
    policy='half_reference' if oldresult['accepted'] and oldresult['selected_val_option']=='half_reference' else 'direct'
    return {'model':model,'scaler':sc,'classes':list(model.classes_),'task':task,'config':cfg,'runtime_blend_policy':policy}

def train():
    fmap=r.v.load_json(r.v.RUN/'folds.json');results={};chosen={};arts={}
    with threadpool_limits(limits=2):
        for task in TASKS:
            rows=r.v.compact_rows(task,'train');classes=sorted({z[r.TARGET[task]] for z in rows});ff=np.array([fmap[z['group_id']] for z in rows])
            folder=OUT/task;folder.mkdir(exist_ok=True);scores=[]
            baseline=r.metric(rows,np.load(r.OUT/task/'current_direct_oof.npy'),classes,r.TARGET[task])
            for cfg in configs():
                x=features(rows,task,cfg,'train');p=np.zeros((len(rows),len(classes)),dtype='float32')
                for fold in range(3):
                    a=np.flatnonzero(ff!=fold);b=np.flatnonzero(ff==fold)
                    assert not {rows[i]['group_id'] for i in a}&{rows[i]['group_id'] for i in b}
                    art=fit(x[a],[rows[i] for i in a],task,cfg);q=art['model'].predict_proba(art['scaler'].transform(x[b]))
                    p[np.ix_(b,[classes.index(c) for c in art['classes']])]=q
                m=r.metric(rows,p,classes,r.TARGET[task]);scores.append({'name':cfg['name'],**m});np.save(folder/(cfg['name']+'_oof.npy'),p)
                print('LETTERBOX_CV',task,cfg['name'],m,flush=True)
            best=max(scores,key=lambda z:z['macro_f1']);gate=best['macro_f1']-baseline['macro_f1']>=.01 and best['accuracy']>=baseline['accuracy']-.01
            cfg=next(c for c in configs() if c['name']==best['name']);art=fit(features(rows,task,cfg,'train'),rows,task,cfg)
            arts[task]=art;joblib.dump(art,folder/'head_train_selected.joblib',compress=3)
            results[task]={'reference':baseline,'candidates':scores,'selected':best,'train_gate':bool(gate),'val_evaluated':False,'accepted':False,'test_used':False}
            r.save(folder/'selection_lock.json',results[task])
            if gate:chosen[task]=cfg
        r.save(OUT/'train_summary.json',results)
        r.save(OUT/'diagnostic_selection_lock.json',{'before_diagnostic_inference':True,'configs':{t:a['config'] for t,a in arts.items()},
            'files_sha256':{t:r.digest(OUT/t/'head_train_selected.joblib') for t in arts},'may_tune_from_external12':False})
        if chosen:extract('val',list(chosen))
        for task,cfg in chosen.items():
            va=r.v.compact_rows(task,'val');art=arts[task];classes=art['classes'];folder=OUT/task
            raw=art['model'].predict_proba(art['scaler'].transform(features(va,task,cfg,'val')))
            new=.5*raw+.5*np.load(r.v.RUN/task/'val_reference.npy') if art['runtime_blend_policy']=='half_reference' else raw
            oldres=r.v.load_json(r.v.RUN/task/'result.json');old=np.load(r.v.RUN/task/('val_selected.npy' if oldres['accepted'] else 'val_reference.npy'))
            before=r.metric(va,old,classes,r.TARGET[task]);after=r.metric(va,new,classes,r.TARGET[task]);ci=r.paired_ci(va,old,new,classes,r.TARGET[task])
            results[task].update({'val_evaluated':True,'val_reference':before,'val_candidate':after,'paired':ci,
                'accepted':bool(after['macro_f1']-before['macro_f1']>=.01 and after['accuracy']>=before['accuracy']-.01 and ci['delta_macro_f1_ci95'][0]>0)})
            np.save(folder/'val_candidate.npy',new)
        for t,z in results.items():r.save(OUT/t/'result.json',z)
        r.save(OUT/'summary.json',results);print('GEOMETRY_RESULTS',results,flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['extract','train']);args=ap.parse_args();setup()
    extract('train') if args.action=='extract' else train()
