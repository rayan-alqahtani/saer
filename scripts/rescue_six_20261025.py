"""تجارب مستقلة مقفلة: تدريب فقط، ثم تحقق استكشافي، بلا اختبار أو تغيير مؤشرات."""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[k]='2'
import sys,json,csv,time,argparse,hashlib
from pathlib import Path
from collections import Counter,defaultdict
import numpy as np
import joblib
from sklearn.model_selection import GroupKFold,StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler,OrdinalEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import accuracy_score,f1_score,classification_report,confusion_matrix
from improve_common import compare_inputs,compact_rows,read_csv,load_json,sha256
from ml_price import PriceModel,PriceFeatures,metrics as price_metrics

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'models/rescue_six_20261025_v1'
OLD=ROOT/'models/improvement_20261021T093545513846'
DUAL=ROOT/'models/improvement_20261024T223322912563/dual_encoder'
BRAND=ROOT/'models/brand_final_20261022T074855'
BASE=ROOT/'models/20261021T090153837145_embeddings'
ACT=ROOT/'models/improvement_20261025T094028750602/activation_cache'
TARGET={'make':'label_make','model':'rec_label','color':'label_color_std','body':'label_body_type_std'}
os.environ['CAR_IMPROVEMENT_RUN']=str(OLD)

def save(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,default=lambda x:x.item() if isinstance(x,np.generic) else str(x)),encoding='utf-8')

def table(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    if not rows:return
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def guard():
    c,tables=compare_inputs()
    assert not c['critical'] and not c['differences']
    assert all(x['authorized_match'] for x in c['files'].values())
    for flag in ['usable_make_recognition','usable_recognition','usable_color','usable_body_type','usable_visual_search']:
        split_by_sha=defaultdict(set)
        for row in tables['images.csv']:
            if row.get(flag)=='True':split_by_sha[row['sha256']].add(row['split'])
        assert not any(len(s)>1 for s in split_by_sha.values()),flag
    protocol=load_json(RUN/'protocol.json')
    for path,digest in protocol['protected'].items():assert sha256(ROOT/path)==digest,path
    return c

def init():
    RUN.mkdir(exist_ok=False)
    c,_=compare_inputs(); assert not c['critical'] and not c['differences']
    protected={str(p.relative_to(ROOT)):sha256(p) for p in (ROOT/'models').glob('*current.json')}
    for p in [BRAND/'brand_model.pt',OLD/'model/best_tail.pt',OLD/'body/best_tail.pt',OLD/'color/best_tail.pt']:
        protected[str(p.relative_to(ROOT))]=sha256(p)
    save(RUN/'protocol.json',{'created':time.time(),'protected':protected,'inputs':c,
      'test_policy':'No test inference or predictions read. Historical validation is exploratory.',
      'price':'5 outer GroupKFold; hierarchical target encoding cross-fit in 3 inner group folds; choose by outer MAE. One selected new config on val.',
      'visual':'3 StratifiedGroupKFold on ImageNet-frozen spatial features only; C .01/.1. Encoder never fitted on these data. One selected configuration per task on val.',
      'search':'One fixed BrandNet cosine candidate, no flip, no tuning. BrandNet selected on historical val; exploratory.',
      'body':'Fixed .75 semantic/.25 direct candidate; .25 candidate/.75 old reference. Infer model for EVERY body image, no dataset membership route.',
      'acceptance':{'price_relative_MAE_gain':.03,'search_top5_gain':.03,'visual_macro_F1_gain':.01,'paired_ci_repetitions':2000,'seed':20261025},
      'visual_fixed_blends':{'make':.25,'model':.25,'color':1.0},
      'no_claim':'No generalization guarantee, no external evaluation; acceptance means development candidate only.'})

def paired(rows,a,b):
    # b-a, statistics supplied per independent group; preserves repeated listings.
    ids=sorted(set(r['group_id'] for r in rows));lookup={g:i for i,g in enumerate(ids)}
    z=np.zeros((len(ids),3))
    for r,x,y in zip(rows,a,b):z[lookup[r['group_id']]] += [1,x,y]
    w=np.random.default_rng(20261025).multinomial(len(ids),np.ones(len(ids))/len(ids),size=2000)
    s=w@z;av=s[:,1]/s[:,0];bv=s[:,2]/s[:,0]
    return {'reference_ci95':np.quantile(av,[.025,.975]).tolist(),'candidate_ci95':np.quantile(bv,[.025,.975]).tolist(),'delta_ci95':np.quantile(bv-av,[.025,.975]).tolist(),'groups':len(ids)}

class Hierarchy:
    keys=[('make',),('make','model_std'),('make','model_std','trim_std')]
    def fit(self,rows):
        self.mean=float(np.mean([np.log(float(r['sar_price'])) for r in rows]));self.maps=[]
        for keys in self.keys:
            d=defaultdict(list)
            for r in rows:d[tuple(r[k] for k in keys)].append(np.log(float(r['sar_price'])))
            self.maps.append({k:(sum(v),len(v)) for k,v in d.items()})
        return self
    def transform(self,rows):
        out=[]
        for r in rows:
            prior=self.mean;v=[]
            for keys,d in zip(self.keys,self.maps):
                s,n=d.get(tuple(r[k] for k in keys),(0,0));prior=(s+5*prior)/(n+5);v.append(prior)
            out.append(v)
        return np.asarray(out)

class HierarchicalPrice:
    def __init__(self,mode='encoded'):self.mode=mode
    def fit(self,rows):
        self.features=PriceFeatures().fit(rows);cats,nums=self.features.transform(rows)
        self.encoder=OrdinalEncoder(handle_unknown='use_encoded_value',unknown_value=np.nan)
        basic=np.column_stack([self.encoder.fit_transform(cats),nums])
        g=np.array([r['group_id'] for r in rows]);te=np.empty((len(rows),3))
        for a,b in GroupKFold(3).split(basic,groups=g):
            assert not set(g[a])&set(g[b])
            te[b]=Hierarchy().fit([rows[i] for i in a]).transform([rows[i] for i in b])
        self.hierarchy=Hierarchy().fit(rows)
        y=np.log([float(r['sar_price']) for r in rows])
        if self.mode=='residual':y=y-te[:,-1]
        self.model=HistGradientBoostingRegressor(loss='absolute_error',learning_rate=.05,max_iter=500,max_leaf_nodes=15,
             min_samples_leaf=10,l2_regularization=1.,early_stopping=False,random_state=42,
             categorical_features=[True]*6+[False]*5)
        self.model.fit(np.column_stack([basic,te]),y);return self
    def predict(self,rows):
        cats,nums=self.features.transform(rows);te=self.hierarchy.transform(rows)
        p=self.model.predict(np.column_stack([self.encoder.transform(cats),nums,te]))
        if self.mode=='residual':p=p+te[:,-1]
        return np.exp(p)

def price():
    guard();folder=RUN/'price';folder.mkdir(exist_ok=False)
    tr=compact_rows('price','train');va=compact_rows('price','val')
    assert len(tr)==1591 and len(va)==357
    cfg={'max_leaf_nodes':15,'min_samples_leaf':10,'max_iter':500,'learning_rate':.05,'l2_regularization':1.}
    factories={'reference':lambda:PriceModel('hgb',cfg),'encoded':lambda:HierarchicalPrice('encoded'),'residual':lambda:HierarchicalPrice('residual')}
    y=np.array([float(r['sar_price']) for r in tr]);g=np.array([r['group_id'] for r in tr]);scores=[]
    for name,factory in factories.items():
        start=time.perf_counter();p=np.empty(len(tr))
        for k,(a,b) in enumerate(GroupKFold(5).split(y,groups=g)):
            assert not set(g[a])&set(g[b]);model=factory().fit([tr[i] for i in a]);p[b]=model.predict([tr[i] for i in b])
            print('PRICE',name,k+1,flush=True)
        np.save(folder/f'{name}_oof.npy',p);scores.append({'candidate':name,**price_metrics(y,p),'seconds':time.perf_counter()-start})
        save(folder/'cv_results.json',scores)
    name=min((s for s in scores if s['candidate']!='reference'),key=lambda s:s['MAE_SAR'])['candidate']
    save(folder/'selection_lock.json',{'selected':name,'selection':'train-only outer group CV','val_not_read':True})
    model=factories[name]().fit(tr);joblib.dump(model,folder/'candidate.joblib',compress=3)
    pointer=load_json(ROOT/'models/price_current.json');ref=joblib.load(ROOT/pointer['run_dir']/'model.joblib')['model']
    a=ref.predict(va);b=model.predict(va);y=np.array([float(r['sar_price']) for r in va]);ci=paired(va,abs(y-a),abs(y-b))
    ma=price_metrics(y,a);mb=price_metrics(y,b)
    accepted=mb['MAE_SAR']<=.97*ma['MAE_SAR'] and ci['delta_ci95'][1]<0 and np.mean(abs(y-b)/y)<=np.mean(abs(y-a)/y)
    result={'reference':ma,'candidate':mb,'paired':ci,'accepted':bool(accepted),'selected':name,'test_used':False,
      'reference_MAPE':float(np.mean(abs(y-a)/y)*100),'candidate_MAPE':float(np.mean(abs(y-b)/y)*100)}
    table(folder/'val_predictions.csv',[{'listing_uid':r['listing_uid'],'group_id':r['group_id'],'actual':float(y[i]),'reference':float(a[i]),'candidate':float(b[i])} for i,r in enumerate(va)])
    counts=Counter(r['make']+'|'+r['model_std'] for r in tr);bands=[]
    for label,lo,hi in [('support_0',0,0),('support_1_4',1,4),('support_5_19',5,19),('support_20plus',20,99999)]:
        mask=np.array([lo<=counts[r['make']+'|'+r['model_std']]<=hi for r in va])
        if mask.any():bands.append({'band':label,'n':int(mask.sum()),'reference_MAE':float(abs(y[mask]-a[mask]).mean()),'candidate_MAE':float(abs(y[mask]-b[mask]).mean())})
    table(folder/'support_bands.csv',bands);save(folder/'result.json',result);guard();print('PRICE_RESULT',result,flush=True)

def rich_color(path):
    from PIL import Image,ImageOps
    import cv2
    with Image.open(path) as im:
        im=ImageOps.exif_transpose(im).convert('RGB');im.thumbnail((160,160));rgb=np.array(im)
    h,w=rgb.shape[:2];out=[]
    for x1,y1,x2,y2 in [(0,0,1,1),(.2,.25,.8,.8),(.3,.35,.7,.7)]:
        crop=rgb[int(y1*h):max(int(y1*h)+1,int(y2*h)),int(x1*w):max(int(x1*w)+1,int(x2*w))]
        hsv=cv2.cvtColor(crop,cv2.COLOR_RGB2HSV).astype(float)/np.array([179,255,255]);lab=cv2.cvtColor(crop,cv2.COLOR_RGB2LAB).astype(float)/255
        flat=np.column_stack([hsv.reshape(-1,3),lab.reshape(-1,3)])
        mask=(flat[:,2]>.12)&~((flat[:,2]>.92)&(flat[:,1]<.10));out.append(float(mask.mean()))
        for a in [flat,flat[mask] if mask.sum()>=20 else flat]:
            out.extend(np.quantile(a,[.1,.25,.5,.75,.9],axis=0).ravel())
            for j in range(6):out.extend(np.histogram(a[:,j],bins=12,range=(0,1))[0]/len(a))
    return np.asarray(out,dtype=np.float32)

def features():
    guard();import torch
    from brand_final_round import BrandNet
    from improve_visual import load_base
    from PIL import Image,ImageOps
    from torchvision.models import ResNet18_Weights
    torch.set_num_threads(2)
    rows=list({r['sha256']:r for t in ['make','model','color','body','search'] for s in ['train','val'] for r in compact_rows(t,s)}.values())
    folder=RUN/'features';folder.mkdir(exist_ok=True)
    if (folder/'index.csv').exists():assert [r['sha256'] for r in read_csv(folder/'index.csv')]==[r['sha256'] for r in rows]
    else:table(folder/'index.csv',[{k:r[k] for k in ['image_uid','sha256','group_id','listing_uid','split','relative_path']} for r in rows])
    idx={r['sha256']:i for i,r in enumerate(read_csv(ACT/'index.csv'))};arr=np.load(ACT/'stage2_0.npy',mmap_mode='r')
    obj=torch.load(BRAND/'brand_model.pt',map_location='cpu',weights_only=True);net=BrandNet(len(obj['classes']));net.load_state_dict(obj['state_dict']);net.eval();b=net.backbone
    mode='r+' if (folder/'spatial.npy').exists() else 'w+'
    space=np.lib.format.open_memmap(folder/'spatial.npy',mode=mode,dtype='float32',shape=(len(rows),512))
    brand=np.lib.format.open_memmap(folder/'brand.npy',mode=mode,dtype='float32',shape=(len(rows),512))
    missing=np.flatnonzero(np.linalg.norm(brand,axis=1)<.5);first=int(missing[0]) if len(missing) else len(rows)
    base=load_base();tf=ResNet18_Weights.IMAGENET1K_V1.transforms()
    start=time.perf_counter()
    with torch.inference_mode():
        for i in range(first,len(rows),24):
            batch=rows[i:i+24]
            if all(r['sha256'] in idx for r in batch):z=torch.from_numpy(np.array(arr[[idx[r['sha256']] for r in batch]],copy=True))
            else:
                xs=[]
                for r in batch:
                    with Image.open(ROOT/'الملفات_المهمة'/r['relative_path']) as im:xs.append(tf(ImageOps.exif_transpose(im).convert('RGB')))
                xx=torch.stack(xs);z=base.layer2(base.layer1(base.maxpool(base.relu(base.bn1(base.conv1(xx))))))
            space[i:i+len(z)]=torch.nn.functional.adaptive_avg_pool2d(z,(2,2)).flatten(1).numpy()
            e=b.avgpool(b.layer4(b.layer3(z))).flatten(1);brand[i:i+len(z)]=torch.nn.functional.normalize(e,dim=1).numpy()
            if i%1200==0:print('FEATURES',i,len(rows),flush=True)
    space.flush();brand.flush();del space,brand,net,arr,base
    colors=[]
    for i,r in enumerate(rows):
        colors.append(rich_color(ROOT/'الملفات_المهمة'/r['relative_path']))
        if i%2000==0:print('COLOR_FEATURES',i,flush=True)
    np.save(folder/'rich_color.npy',np.stack(colors));save(folder/'metadata.json',{'seconds':time.perf_counter()-start,'n':len(rows),'test_images':0,'spatial_encoder':'frozen ImageNet layer2 2x2 pool','brand_encoder':str(BRAND),'rich_color_dimensions':len(colors[0])})
    guard()

def x_for(rows,kind):
    index={r['sha256']:i for i,r in enumerate(read_csv(RUN/'features/index.csv'))};pos=[index[r['sha256']] for r in rows]
    bi={r['sha256']:i for i,r in enumerate(read_csv(BASE/'image_index.csv'))};bp=[bi[r['sha256']] for r in rows]
    e=np.load(BASE/'embeddings.npy',mmap_mode='r')[bp]
    if kind=='brand':return np.asarray(np.load(RUN/'features/brand.npy',mmap_mode='r')[pos])
    c=np.load(BASE/'color_features.npy',mmap_mode='r')[bp]
    if kind=='color':return np.column_stack([e,c,np.load(RUN/'features/rich_color.npy',mmap_mode='r')[pos]])
    return np.column_stack([e,np.load(RUN/'features/spatial.npy',mmap_mode='r')[pos]])

def align_saved(folder,rows,name='val_probabilities.npy'):
    rr=read_csv(folder/'val_predictions.csv');idx={r['image_uid']:i for i,r in enumerate(rr)};p=np.load(folder/name)
    return p[[idx[r['image_uid']] for r in rows]]

def f1_ci(rows,y,a,b,classes):
    # Cluster bootstrap stratified by true class only when every group is class-pure.
    groups=defaultdict(list)
    for i,r in enumerate(rows):groups[r['group_id']].append(i)
    pure=all(len(set(y[i] for i in ids))==1 for ids in groups.values());strata=defaultdict(list)
    for g,ids in groups.items():strata[y[ids[0]] if pure else 'all'].append(g)
    rng=np.random.default_rng(20261025);gkeys=list(groups);gi={g:i for i,g in enumerate(gkeys)};cl={c:i for i,c in enumerate(classes)}
    n=len(classes);true=np.zeros((len(gkeys),n));pa=true.copy();pb=true.copy();ha=true.copy();hb=true.copy()
    for g,ids in groups.items():
        for i in ids:
            k=gi[g];true[k,cl[y[i]]]+=1;pa[k,cl[a[i]]]+=1;pb[k,cl[b[i]]]+=1
            ha[k,cl[y[i]]]+=a[i]==y[i];hb[k,cl[y[i]]]+=b[i]==y[i]
    w=np.zeros((2000,len(gkeys)))
    for gs in strata.values():w[:,[gi[g] for g in gs]]=rng.multinomial(len(gs),np.ones(len(gs))/len(gs),size=2000)
    t=w@true;fa=np.divide(2*(w@ha),t+w@pa,out=np.zeros_like(t),where=(t+w@pa)>0).mean(1)
    fb=np.divide(2*(w@hb),t+w@pb,out=np.zeros_like(t),where=(t+w@pb)>0).mean(1)
    return {'delta_ci95':np.quantile(fb-fa,[.025,.975]).tolist(),'candidate_ci95':np.quantile(fb,[.025,.975]).tolist(),'groups':len(groups),'class_stratified':pure,'conditional_on_observed_class_support':pure,'repetitions':2000}

def classification_result(task,rows,classes,old,new,extra=None):
    folder=RUN/task;folder.mkdir(exist_ok=True);y=np.array([r[TARGET[task]] for r in rows]);a=np.array(classes)[old.argmax(1)];b=np.array(classes)[new.argmax(1)]
    fn=lambda p:{'accuracy':float(accuracy_score(y,p)),'macro_F1':float(f1_score(y,p,labels=classes,average='macro',zero_division=0)),'n':len(y)}
    ma,mb=fn(a),fn(b);ci=f1_ci(rows,y,a,b,classes)
    accepted=mb['macro_F1']-ma['macro_F1']>=.01 and ci['delta_ci95'][0]>0
    result={'reference':ma,'candidate':mb,'paired':ci,'accepted':bool(accepted),'test_used':False,**(extra or {})}
    save(folder/'result.json',result);np.save(folder/'val_candidate.npy',new)
    table(folder/'val_predictions.csv',[{'image_uid':r['image_uid'],'group_id':r['group_id'],'actual':y[i],'reference':a[i],'candidate':b[i]} for i,r in enumerate(rows)])
    report=classification_report(y,b,labels=classes,output_dict=True,zero_division=0)
    table(folder/'val_per_class.csv',[{'class':c,**report[c]} for c in classes]);save(folder/'val_confusion.json',{'classes':classes,'matrix':confusion_matrix(y,b,labels=classes).tolist()})
    table(folder/'worst10.csv',[{'class':c,**report[c],'reason':'يحتاج تشخيصا؛ الدعم أو التشابه فرضية وليست سببا مثبتا'} for c in sorted(classes,key=lambda c:report[c]['f1-score'])[:10]])
    print('VISUAL_RESULT',task,result,flush=True)

def visual(task):
    guard();folder=RUN/task;folder.mkdir(exist_ok=False)
    tr=compact_rows(task,'train');va=compact_rows(task,'val');y=np.array([r[TARGET[task]] for r in tr]);g=np.array([r['group_id'] for r in tr]);classes=sorted(set(y))
    x=x_for(tr,'color' if task=='color' else 'spatial');scores=[]
    for c in [.01,.1]:
        pred=np.empty(len(y),dtype=object);start=time.perf_counter()
        for fold,(a,b) in enumerate(StratifiedGroupKFold(3,shuffle=True,random_state=20261025).split(x,y,g)):
            assert not set(g[a])&set(g[b]);sc=StandardScaler().fit(x[a]);counts=Counter(g[a]);weights=np.array([1/counts[v] for v in g[a]]);weights/=weights.mean()
            model=LogisticRegression(C=c,class_weight='balanced',max_iter=500,solver='lbfgs').fit(sc.transform(x[a]),y[a],sample_weight=weights)
            pred[b]=model.predict(sc.transform(x[b]));print('HEAD',task,c,fold+1,flush=True)
        scores.append({'C':c,'macro_F1':float(f1_score(y,pred,labels=classes,average='macro',zero_division=0)),'seconds':time.perf_counter()-start})
        save(folder/'cv_results.json',scores)
    best=max(scores,key=lambda r:r['macro_F1']);save(folder/'selection_lock.json',{'config':best,'fit':'train-only CV, ImageNet frozen','val_not_used':True})
    sc=StandardScaler().fit(x);counts=Counter(g);weights=np.array([1/counts[v] for v in g]);weights/=weights.mean()
    model=LogisticRegression(C=best['C'],class_weight='balanced',max_iter=500,solver='lbfgs').fit(sc.transform(x),y,sample_weight=weights)
    assert list(model.classes_)==classes
    art={'model':model,'scaler':sc,'classes':classes,'feature_kind':'color' if task=='color' else 'spatial'};joblib.dump(art,folder/'head.joblib',compress=3)
    p=model.predict_proba(sc.transform(x_for(va,art['feature_kind'])))
    if task=='make':
        assert classes==load_json(BRAND/'protocol.json')['classes']
        ref=align_saved(BRAND,va)
    elif task=='model':
        assert classes==load_json(OLD/'model/selected.json')['classes']
        ref=np.load(DUAL/'model/val_best_blend_probabilities.npy')
        # Dual training uses exactly compact_rows order, check saved old reference row membership.
        assert ref.shape==p.shape
    else:
        assert classes==load_json(OLD/task/'selected.json')['classes']
        ref=align_saved(OLD/task,va)
    w=1. if task=='color' else .25
    classification_result(task,va,classes,ref,(1-w)*ref+w*p,{'fixed_new_weight':w,'direct_head_macro_F1':float(f1_score([r[TARGET[task]] for r in va],np.array(classes)[p.argmax(1)],average='macro',zero_division=0)),'encoder_fit':'ImageNet only for new head; blend reference selected historically on val'})
    guard()

def search():
    guard();folder=RUN/'search';folder.mkdir(exist_ok=False)
    tr=compact_rows('search','train');first={}
    for r in compact_rows('search','val'):first.setdefault(r['group_id'],r)
    va=list(first.values());e=x_for(tr,'brand');q=x_for(va,'brand');saved=read_csv(OLD/'search/val_predictions.csv');idx={r['query_uid']:r for r in saved}
    records=[]
    for n,r in enumerate(va):
        sims=e@q[n];found=[];seen=set()
        for i in np.argsort(-sims):
            t=tr[i]
            if t['group_id']==r['group_id'] or t['listing_uid']==r['listing_uid'] or t['sha256']==r['sha256'] or t['group_id'] in seen:continue
            seen.add(t['group_id']);found.append(t)
            if len(found)==5:break
        assert len(found)==5
        assert all(t['group_id']!=r['group_id'] and t['listing_uid']!=r['listing_uid'] and t['sha256']!=r['sha256'] for t in found)
        old=idx[r['image_uid']];h=[t['rec_label']==r['rec_label'] for t in found]
        records.append({'query_uid':r['image_uid'],'group_id':r['group_id'],'reference_top1':float(old['top1_hit']),'reference_top5':float(old['top5_hit']),'candidate_top1':int(h[0]),'candidate_top5':int(any(h)),'precision5':sum(h)/5,'retrieved_listing_uids':';'.join(t['listing_uid'] for t in found)})
    result={'n':len(va),'test_used':False,'excluded_leakage_assertions_passed':True,'encoder':'BrandNet original view, no TTA, no whitening'}
    for metric in ['top1','top5']:
        a=[r['reference_'+metric] for r in records];b=[r['candidate_'+metric] for r in records]
        result[metric]={'reference':float(np.mean(a)),'candidate':float(np.mean(b)),**paired(va,a,b)}
    result['accepted']=bool(result['top5']['candidate']-result['top5']['reference']>=.03 and result['top5']['delta_ci95'][0]>0 and result['top1']['candidate']>=result['top1']['reference'])
    result['precision5']=float(np.mean([r['precision5'] for r in records]));table(folder/'val_predictions.csv',records)
    np.save(folder/'gallery_embeddings.npy',e);table(folder/'gallery.csv',[{k:r[k] for k in ['image_uid','listing_uid','group_id','sha256','relative_path','rec_label']} for r in tr])
    save(folder/'result.json',result);guard();print('SEARCH_RESULT',result,flush=True)

def body():
    guard();import torch
    from retrain_dual_encoder_six import probabilities
    va=compact_rows('body','val');trb=compact_rows('body','train');trm=compact_rows('model','train')
    art=joblib.load(DUAL/'model/head.joblib');ba=joblib.load(DUAL/'body/head.joblib')
    bi={r['sha256']:i for i,r in enumerate(read_csv(BASE/'image_index.csv'))};bp=[bi[r['sha256']] for r in va]
    x=np.column_stack([np.load(BASE/'embeddings.npy',mmap_mode='r')[bp],x_for(va,'brand')])
    pm=probabilities(art['model'],art['scaler'],x,art['classes'],art['tau']);pb=probabilities(ba['model'],ba['scaler'],x,ba['classes'],ba['tau'])
    mc=art['classes'];bc=ba['classes'];lookup={r['image_uid']:r for r in trm};counts=np.ones((len(mc),len(bc)))
    for r in trb:
        if r['image_uid'] in lookup:counts[mc.index(lookup[r['image_uid']]['rec_label']),bc.index(r['label_body_type_std'])]+=1
    counts/=counts.sum(1,keepdims=True)
    candidate=.25*pb+.75*(pm@counts);ref=align_saved(OLD/'body',va)
    classification_result('body',va,bc,ref,.75*ref+.25*candidate,{'fix':'All images pass through model head; no membership or labels in routing','route_uses_metadata':False,'promotion_requires_gate':True})
    np.save(RUN/'body/transition.npy',counts);guard()

def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['init','price','features','make','model','color','body','search']);a=p.parse_args()
    if a.command in ['make','model','color']:visual(a.command)
    else:globals()[a.command]()

if __name__=='__main__':
    sys.modules['rescue_six_20261025']=sys.modules[__name__]
    for cls in [Hierarchy,HierarchicalPrice]:cls.__module__='rescue_six_20261025'
    main()
