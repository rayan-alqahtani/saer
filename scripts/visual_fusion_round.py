"""تجارب DINOv2 وقص السيارة: طيات train، تحقق استكشافي، لا test."""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'): os.environ[k]='2'
import argparse, csv, json, time, sys, shutil
from pathlib import Path
from collections import Counter,defaultdict
import numpy as np
import joblib
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score,f1_score,classification_report,confusion_matrix
import rescue_six_20261025 as old
from improve_common import compact_rows,read_csv,load_json,sha256
ROOT=old.ROOT
RUN=ROOT/'models/visual_fusion_20261025_v1'
TASKS=['make','model','color','body']
SEED=20261025
HUB=Path.home()/'.cache/torch/hub'

def save(p,o):old.save(p,o)
def table(p,r):old.table(p,r)
def guard():
    c=old.guard()
    if (RUN/'protocol.json').exists():
        for p,h in load_json(RUN/'protocol.json')['protected'].items():assert sha256(Path(p))==h,p
    return c

def init():
    c=guard();RUN.mkdir(exist_ok=False);(RUN/'features').mkdir()
    rr=list({r['sha256']:r for t in TASKS+['search'] for s in ['train','val'] for r in compact_rows(t,s)}.values())
    assert all(r['split'] in ('train','val') for r in rr)
    table(RUN/'features/index.csv',rr)
    gs=sorted({r['group_id'] for r in rr if r['split']=='train'})
    folds={}
    for fold,(_,b) in enumerate(GroupKFold(3).split(gs,groups=gs)):
        for i in b:folds[gs[i]]=fold
    save(RUN/'folds.json',folds)
    protect=list((ROOT/'models').glob('*current.json'))+[HUB/'checkpoints/dinov2_vits14_pretrain.pth',HUB/'checkpoints/ssdlite320_mobilenet_v3_large_coco-a79551df.pth',old.RUN/'final_lock.json']
    save(RUN/'protocol.json',{'seed':SEED,'inputs':c,'protected':{str(p):sha256(p) for p in protect},
       'training':'Frozen pretrained DINOv2 ViT-S/14 + frozen ImageNet ResNet18; new supervised heads trained from scratch. Equal total group weight; balanced classes.',
       'candidates':'DINO CLS, DINO+ResNet, cropped DINO+ResNet, full+crop DINO+ResNet; C=.1,1. Color adds original/cropped HSV/Lab features. Select by 3 train group folds only.',
       'crop':'COCO SSDLite, car score>=.4; largest score*sqrt(area), margin5%; min80x60; fallback original. No logo detector or paint segmentation claim.',
       'preprocess':'EXIF transpose RGB; DINO resize256 bicubic center224 ImageNet norm; original files immutable; resized input does not increase real image detail.',
       'val_candidates':'Two predeclared options per task: train-CV selected head alone and equal-weight blend with strongest retained reference. Selection exploratory on historical val.',
       'acceptance':'macro-F1 gain>=.01 and positive paired group Bayesian-bootstrap 95% CI; search top5 gain>=.03 positive paired CI and no top1 decline.',
       'bootstrap':'2000 exponential independent group weights, normalized implicitly in metrics; all observed classes retained; Bayesian bootstrap conditional on observed groups, not selection-adjusted.',
       'test':'Never infer test or read test predictions. Historical val reused; no external generalization proof.',
       'secondary':'Listing probability averaging reported separately, only listings with consistent task labels; no source labels altered.',
       'price':'Unchanged; latest accepted tabular price is rescue_six_20261025_v1.',
       'search':'Choose frozen DINO/fullcrop/ResNet mixtures by train group retrieval CV with fixed first image per group, then one val candidate against current BrandNet search.',
       'hierarchy':'Optional make/model consistency uses predicted probabilities, never true model membership; fixed threshold .7 and weight .5, exploratory additional val candidate.',
       'source':sha256(Path(__file__))})
    table(RUN/'counts.csv',[{'task':t,'split':s,'images':len(rs),'listings':len({r['listing_uid'] for r in rs}),'groups':len({r['group_id'] for r in rs})} for t in TASKS+['search'] for s in ['train','val'] for rs in [compact_rows(t,s)]])
    shutil.copy2(ROOT/'PROGRESS.md',RUN/'PROGRESS_before.md')
    print('INIT',len(rr),flush=True)

def models():
    import torch
    from torchvision import transforms
    from torchvision.models.detection import ssdlite320_mobilenet_v3_large,SSDLite320_MobileNet_V3_Large_Weights
    torch.set_num_threads(int(os.environ.get('CAR_VISUAL_THREADS','6')))
    d=torch.hub.load(str(HUB/'facebookresearch_dinov2_main'),'dinov2_vits14',source='local',pretrained=False)
    d.load_state_dict(torch.load(HUB/'checkpoints/dinov2_vits14_pretrain.pth',map_location='cpu',weights_only=True));d.eval().requires_grad_(False)
    detector=ssdlite320_mobilenet_v3_large(weights=SSDLite320_MobileNet_V3_Large_Weights.DEFAULT).eval().requires_grad_(False)
    tf=transforms.Compose([transforms.Resize(256,interpolation=transforms.InterpolationMode.BICUBIC),transforms.CenterCrop(224),transforms.ToTensor(),transforms.Normalize((.485,.456,.406),(.229,.224,.225))])
    return d,detector,tf

def crop_image(im,pred):
    w,h=im.size;cs=[]
    for b,l,s in zip(pred['boxes'],pred['labels'],pred['scores']):
        b=b.tolist();s=float(s)
        if int(l)==3 and s>=.4:
            cs.append((s*np.sqrt(max(0,(b[2]-b[0])*(b[3]-b[1]))/(w*h)),s,b))
    if not cs:return im,[0,0,w,h],0.,False
    _,score,b=max(cs);x1,y1,x2,y2=b;dx=(x2-x1)*.05;dy=(y2-y1)*.05
    box=[max(0,int(x1-dx)),max(0,int(y1-dy)),min(w,int(x2+dx)),min(h,int(y2+dy))]
    if box[2]-box[0]<80 or box[3]-box[1]<60:return im,[0,0,w,h],score,False
    return im.crop(box),box,score,True

def color_desc(im):
    import cv2
    cv2.setNumThreads(1)
    im=im.copy();im.thumbnail((160,160));rgb=np.array(im);h,w=rgb.shape[:2];out=[]
    for x1,y1,x2,y2 in [(0,0,1,1),(.15,.35,.85,.85),(.3,.45,.7,.75)]:
        a=rgb[int(y1*h):max(int(y1*h)+1,int(y2*h)),int(x1*w):max(int(x1*w)+1,int(x2*w))]
        hsv=cv2.cvtColor(a,cv2.COLOR_RGB2HSV)/np.array([179,255,255]);lab=cv2.cvtColor(a,cv2.COLOR_RGB2LAB)/255.
        z=np.column_stack([hsv.reshape(-1,3),lab.reshape(-1,3)])
        out.extend(np.quantile(z,[.1,.25,.5,.75,.9],axis=0).ravel())
        for j in range(6):out.extend(np.histogram(z[:,j],bins=12,range=(0,1))[0]/len(z))
    return np.asarray(out,dtype=np.float32)

def extract():
    guard();import torch
    from PIL import Image,ImageOps
    from torchvision.transforms.functional import to_tensor
    rr=read_csv(RUN/'features/index.csv');n=len(rr);folder=RUN/'features'
    arrays={}
    for name,dim in [('full',384),('crop',384),('colors',306)]:
        p=folder/(name+'.npy');arrays[name]=np.lib.format.open_memmap(p,mode='r+' if p.exists() else 'w+',dtype='float32',shape=(n,dim))
    prior=ROOT/'models/image_price_dinov2_multiview_20261024'
    pr=read_csv(prior/'train_images.csv');pe=np.load(prior/'cache/train_cls_all_views.npy',mmap_mode='r');pi={r['sha256']:i for i,r in enumerate(pr)}
    done=load_json(folder/'progress.json')['done'] if (folder/'progress.json').exists() else 0
    d,det,tf=models();t=time.perf_counter();boxes=read_csv(folder/'boxes.csv') if (folder/'boxes.csv').exists() else []
    boxes=boxes[:done]
    with torch.inference_mode():
        for st in range(done,n,8):
            batch=rr[st:st+8];ims=[]
            for r in batch:
                p=ROOT/'الملفات_المهمة'/r['relative_path'];assert sha256(p)==r['sha256']
                with Image.open(p) as src:ims.append(ImageOps.exif_transpose(src).convert('RGB'))
            pp=det([to_tensor(im) for im in ims]);crops=[];fresh=[];ids=[]
            for j,(r,im,p) in enumerate(zip(batch,ims,pp)):
                crop,box,score,found=crop_image(im,p);crops.append(tf(crop));arrays['colors'][st+j]=color_desc(crop)
                boxes.append({'image_uid':r['image_uid'],'sha256':r['sha256'],'box':json.dumps(box),'score':score,'found':found})
                if r['sha256'] in pi:arrays['full'][st+j]=pe[pi[r['sha256']]]
                else:fresh.append(tf(im));ids.append(st+j)
            if fresh:arrays['full'][ids]=d(torch.stack(fresh)).numpy()
            arrays['crop'][st:st+len(batch)]=d(torch.stack(crops)).numpy()
            if st%160==0 or st+len(batch)==n:
                for a in arrays.values():a.flush()
                table(folder/'boxes.csv',boxes);save(folder/'progress.json',{'done':st+len(batch),'total':n,'seconds_current_invocation':time.perf_counter()-t})
                print('FEATURES',st+len(batch),'/',n,'seconds',round(time.perf_counter()-t),flush=True)
    table(folder/'boxes.csv',boxes);save(folder/'progress.json',{'done':n,'total':n,'complete':True,'reused_full_features':sum(r['sha256'] in pi for r in rr),'detected':sum(str(r['found'])=='True' for r in boxes)})
    guard()

def feats(rows,kind,task=None):
    idx={r['sha256']:i for i,r in enumerate(read_csv(RUN/'features/index.csv'))};ii=[idx[r['sha256']] for r in rows]
    full=np.load(RUN/'features/full.npy',mmap_mode='r')[ii];crop=np.load(RUN/'features/crop.npy',mmap_mode='r')[ii]
    bi={r['sha256']:i for i,r in enumerate(read_csv(old.BASE/'image_index.csv'))};bp=[bi[r['sha256']] for r in rows]
    base=np.load(old.BASE/'embeddings.npy',mmap_mode='r')[bp]
    if kind=='color_only':return np.column_stack([np.load(old.BASE/'color_features.npy',mmap_mode='r')[bp],np.load(RUN/'features/colors.npy',mmap_mode='r')[ii]]).astype('float32')
    parts={'dino':[full],'fusion':[full,base],'crop':[crop,base],'both':[full,crop,base]}[kind]
    if task=='color':parts=parts+[np.load(old.BASE/'color_features.npy',mmap_mode='r')[bp]]+([np.load(RUN/'features/colors.npy',mmap_mode='r')[ii]] if kind in ('crop','both') else [])
    return np.column_stack(parts).astype('float32')

def fit(x,rows,c):
    g=[r['group_id'] for r in rows];cnt=Counter(g);w=np.array([1/cnt[v] for v in g]);w/=w.mean()
    sc=StandardScaler().fit(x)
    m=HistGradientBoostingClassifier(max_iter=180,learning_rate=.05,max_leaf_nodes=15,min_samples_leaf=15,l2_regularization=2.,class_weight='balanced',early_stopping=False,random_state=SEED) if c==-1 else LogisticRegression(C=c,class_weight='balanced',max_iter=600,random_state=SEED)
    m.fit(sc.transform(x),[r['_target'] for r in rows],sample_weight=w)
    return sc,m

def met(y,p,classes):
    h=np.array(classes)[p.argmax(1)]
    return {'accuracy':float(accuracy_score(y,h)),'macro_F1':float(f1_score(y,h,labels=classes,average='macro',zero_division=0)),'n':len(y)}

def ci(rows,y,a,b,classes):
    gids=sorted({r['group_id'] for r in rows});gi={g:i for i,g in enumerate(gids)};cl={c:i for i,c in enumerate(classes)};shape=(len(gids),len(classes))
    t=np.zeros(shape);pa=t.copy();pb=t.copy();ha=t.copy();hb=t.copy()
    for r,v,x,z in zip(rows,y,a,b):
        i=gi[r['group_id']];t[i,cl[v]]+=1;pa[i,cl[x]]+=1;pb[i,cl[z]]+=1;ha[i,cl[v]]+=v==x;hb[i,cl[v]]+=v==z
    w=np.random.default_rng(SEED).exponential(size=(2000,len(gids)))
    tt=w@t;aa=2*(w@ha)/(tt+w@pa);bb=2*(w@hb)/(tt+w@pb)
    fa=aa.mean(1);fb=bb.mean(1)
    return {'delta_macro_F1':np.quantile(fb-fa,[.025,.975]).tolist(),'candidate_macro_F1':np.quantile(fb,[.025,.975]).tolist(),'groups':len(gids),'method':'group Bayesian bootstrap 2000; conditional on observed support'}

def reference(task,rows):
    if task=='make':return old.align_saved(old.BRAND,rows)
    if task=='model':
        assert [r['image_uid'] for r in read_csv(old.RUN/'model/val_predictions.csv')]==[r['image_uid'] for r in rows]
        return np.load(old.DUAL/'model/val_best_blend_probabilities.npy')
    return old.align_saved(old.OLD/task,rows)

def listing_metrics(rows,y,p,classes):
    groups=defaultdict(list)
    for i,r in enumerate(rows):groups[r['listing_uid']].append(i)
    yy=[];pp=[];bad=0
    for ids in groups.values():
        if len(set(y[i] for i in ids))!=1:bad+=1;continue
        yy.append(y[ids[0]]);pp.append(p[ids].mean(0))
    return {**met(yy,np.array(pp),classes),'conflicting_listings_excluded_from_secondary_metric':bad,'metric_unit':'listing, probability mean across eligible images'}

def classify(task):
    guard();folder=RUN/task;folder.mkdir(exist_ok=True)
    tr=compact_rows(task,'train');target=old.TARGET[task]
    for r in tr:r['_target']=r[target]
    classes=sorted({r[target] for r in tr});y=np.array([r[target] for r in tr]);folds=load_json(RUN/'folds.json');ff=np.array([folds[r['group_id']] for r in tr]);trials=[]
    for kind in ['dino','fusion','crop','both']+(['color_only'] if task=='color' else []):
        x=feats(tr,kind,task)
        for c in ([-1] if kind=='color_only' else [.1,1.]):
            trial_start=time.perf_counter()
            dest=folder/f'oof_{kind}_{c}.npy'
            if dest.exists():p=np.load(dest)
            else:
                p=np.zeros((len(tr),len(classes)),dtype='float32')
                for fold in range(3):
                    a=np.where(ff!=fold)[0];b=np.where(ff==fold)[0];sc,m=fit(x[a],[tr[i] for i in a],c)
                    ix=[classes.index(v) for v in m.classes_];p[np.ix_(b,ix)]=m.predict_proba(sc.transform(x[b]));print('CV',task,kind,c,fold,flush=True)
                np.save(dest,p)
            trials.append({'kind':kind,'C':c,**met(y,p,classes),'seconds':time.perf_counter()-trial_start});table(folder/'train_cv.csv',trials)
    best=max(trials,key=lambda r:r['macro_F1']);save(folder/'selection_lock.json',best)
    sc,m=fit(feats(tr,best['kind'],task),tr,best['C']);assert list(m.classes_)==classes
    save(folder/'estimator_params.json',{'head':type(m).__name__,'parameters':m.get_params(deep=False),'scaler':sc.get_params(deep=False),'classes':classes,'fit_split':'train','backbones_frozen':True})
    art={'scaler':sc,'model':m,'classes':classes,'kind':best['kind'],'task':task,'selection':best};joblib.dump(art,folder/'head.joblib',compress=3)
    va=compact_rows(task,'val');yv=np.array([r[target] for r in va]);p=m.predict_proba(sc.transform(feats(va,best['kind'],task)));ref=reference(task,va)
    assert p.shape==ref.shape
    np.save(folder/'val_direct.npy',p);np.save(folder/'val_reference.npy',ref)
    opts=[('direct',p),('half_reference',.5*p+.5*ref)]
    scores=[{'name':name,**met(yv,q,classes)} for name,q in opts];table(folder/'val_options.csv',scores)
    name,q=max(opts,key=lambda v:met(yv,v[1],classes)['macro_F1']);finish_task(task,va,classes,ref,q,name,scores)
    guard()

def finish_task(task,rows,classes,ref,p,name,scores):
    folder=RUN/task;y=np.array([r[old.TARGET[task]] for r in rows]);a=np.array(classes)[ref.argmax(1)];b=np.array(classes)[p.argmax(1)];before=met(y,ref,classes);after=met(y,p,classes);interval=ci(rows,y,a,b,classes)
    result={'reference':before,'candidate':after,'paired':interval,'accepted':bool(after['macro_F1']-before['macro_F1']>=.01 and interval['delta_macro_F1'][0]>0),'selected_val_option':name,'val_options':scores,'test_used':False,'listing_reference':listing_metrics(rows,y,ref,classes),'listing_candidate':listing_metrics(rows,y,p,classes)}
    save(folder/'result.json',result);np.save(folder/'val_selected.npy',p)
    table(folder/'val_predictions.csv',[{'image_uid':r['image_uid'],'listing_uid':r['listing_uid'],'group_id':r['group_id'],'actual':y[i],'reference':a[i],'candidate':b[i]} for i,r in enumerate(rows)])
    report=classification_report(y,b,labels=classes,output_dict=True,zero_division=0)
    train_rows=compact_rows(task,'train');support={c:{'train_images':sum(r[old.TARGET[task]]==c for r in train_rows),'train_groups':len({r['group_id'] for r in train_rows if r[old.TARGET[task]]==c})} for c in classes}
    table(folder/'per_class.csv',[{'class':c,**report[c],**support[c]} for c in classes]);save(folder/'confusion.json',{'classes':classes,'matrix':confusion_matrix(y,b,labels=classes).tolist()})
    table(folder/'worst10.csv',[{'class':c,**report[c],**support[c],'cause':'السبب غير مثبت؛ يلزم فحص مستقل للوسم والزاوية والدعم'} for c in sorted(classes,key=lambda c:report[c]['f1-score'])[:10]])
    print('RESULT',task,json.dumps(result),flush=True)

def hierarchy():
    guard();rows=compact_rows('make','val');art=joblib.load(RUN/'model/head.joblib');make=joblib.load(RUN/'make/head.joblib');p=art['model'].predict_proba(art['scaler'].transform(feats(rows,art['kind'],'model')))
    trans=np.zeros((len(art['classes']),len(make['classes'])))
    tr=compact_rows('model','train');mapping=defaultdict(set)
    for r in tr:mapping[r['rec_label']].add(r['label_make'])
    for j,c in enumerate(art['classes']):
        assert len(mapping[c])==1;trans[j,make['classes'].index(next(iter(mapping[c])))]=1
    ref=np.load(RUN/'make/val_reference.npy');direct=np.load(RUN/'make/val_direct.npy');new=np.where((p.max(1)>=.7)[:,None],.5*direct+.5*(p@trans),direct)
    rowscores=read_csv(RUN/'make/val_options.csv');yv=[r['label_make'] for r in rows];score={'name':'hierarchy_fixed',**met(yv,new,make['classes'])};rowscores.append(score)
    # All three candidates declared before results; historical val is exploratory.
    previous=load_json(RUN/'make/result.json')
    save(RUN/'make/before_hierarchy.json',previous);np.save(RUN/'make/model_to_make.npy',trans)
    if score['macro_F1']>previous['candidate']['macro_F1']:
        backup=RUN/'make/before_hierarchy';backup.mkdir(exist_ok=False)
        for file in ['result.json','val_selected.npy','val_predictions.csv','per_class.csv','confusion.json','worst10.csv']:
            shutil.copy2(RUN/'make'/file,backup/file)
        finish_task('make',rows,make['classes'],ref,new,'hierarchy_fixed',rowscores)
    table(RUN/'make/val_options.csv',rowscores);save(RUN/'make/hierarchy_trial.json',score)

def norm(x):return x/np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-12)
def search_feats(rows,kind):
    x=feats(rows,'both');full=norm(x[:,:384]);crop=norm(x[:,384:768]);base=norm(x[:,768:])
    if kind=='dino':return full
    if kind=='fusion':return norm(np.column_stack([full,base]))
    return norm(np.column_stack([full,crop,base]))

def hits(qrows,grows,q,g):
    out=[]
    for st in range(0,len(q),32):
        sims=q[st:st+32]@g.T
        for k,s in enumerate(sims):
            r=qrows[st+k];seen=set();found=[]
            for i in np.argsort(-s):
                v=grows[i]
                if v['group_id']==r['group_id'] or v['listing_uid']==r['listing_uid'] or v['sha256']==r['sha256'] or v['group_id'] in seen:continue
                seen.add(v['group_id']);found.append(v)
                if len(found)==5:break
            hh=[v['rec_label']==r['rec_label'] for v in found]
            out.append({'query_uid':r['image_uid'],'group_id':r['group_id'],'top1':int(hh[0]),'top5':int(any(hh)),'precision5':sum(hh)/5,'retrieved':';'.join(v['listing_uid'] for v in found)})
    return out

def search():
    guard();folder=RUN/'search';folder.mkdir(exist_ok=True);tr=compact_rows('search','train');folds=load_json(RUN/'folds.json');scores=[]
    for kind in ['dino','fusion','both','both_whiten']:
        x=search_feats(tr,'both' if kind=='both_whiten' else kind);records=[]
        for fold in range(3):
            a=[i for i,r in enumerate(tr) if folds[r['group_id']]!=fold];first={}
            for i,r in enumerate(tr):
                if folds[r['group_id']]==fold:first.setdefault(r['group_id'],i)
            b=list(first.values());xx=x[a];qq=x[b]
            if kind=='both_whiten':
                pca=PCA(n_components=128,whiten=True,svd_solver='randomized',random_state=SEED).fit(xx);qq=norm(pca.transform(qq));xx=norm(pca.transform(xx))
            records+=hits([tr[i] for i in b],[tr[i] for i in a],qq,xx)
        scores.append({'kind':kind,'top1':float(np.mean([r['top1'] for r in records])),'top5':float(np.mean([r['top5'] for r in records])),'n':len(records)})
        table(folder/'train_cv.csv',scores);print('SEARCH_CV',scores[-1],flush=True)
    best=max(scores,key=lambda r:(r['top5'],r['top1']));save(folder/'selection_lock.json',best)
    first={}
    for r in compact_rows('search','val'):first.setdefault(r['group_id'],r)
    va=list(first.values());kind=best['kind'];g=search_feats(tr,'both' if kind=='both_whiten' else kind);q=search_feats(va,'both' if kind=='both_whiten' else kind)
    if kind=='both_whiten':
        pca=PCA(n_components=128,whiten=True,svd_solver='randomized',random_state=SEED).fit(g);joblib.dump(pca,folder/'pca.joblib');q=norm(pca.transform(q));g=norm(pca.transform(g))
    records=hits(va,tr,q,g)
    prior={r['query_uid']:r for r in read_csv(old.RUN/'search/val_predictions.csv')};result={'n':len(va),'selected':best,'test_used':False}
    for metric in ['top1','top5']:
        a=[float(prior[r['query_uid']]['candidate_'+metric]) for r in records];b=[r[metric] for r in records]
        result[metric]={'reference':float(np.mean(a)),'candidate':float(np.mean(b)),**old.paired(va,a,b)}
        for r,v in zip(records,a):r['reference_'+metric]=v
    result['accepted']=bool(result['top5']['candidate']-result['top5']['reference']>=.03 and result['top5']['delta_ci95'][0]>0 and result['top1']['candidate']>=result['top1']['reference'])
    result['precision5']=float(np.mean([r['precision5'] for r in records]));save(folder/'result.json',result);table(folder/'val_predictions.csv',records);table(folder/'gallery.csv',tr);np.save(folder/'gallery.npy',g)
    print('SEARCH_RESULT',json.dumps(result),flush=True);guard()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['init','extract','make','model','color','body','hierarchy','search']);a=p.parse_args()
    if a.command in TASKS:classify(a.command)
    else:globals()[a.command]()
