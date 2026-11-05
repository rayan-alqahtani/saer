"""إعادة تدريب رؤوس بصرية مستقلة على تمثيلات ResNet18 + BrandNet المجمدة.

لا يقرأ test أثناء اختيار الرؤوس. يُستخرج test بعد قفل الإعدادات مرة واحدة.
"""
from improve_common import *
from ml_visual import embedding_data, color_features
from brand_final_round import BrandNet
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.model_selection import GroupKFold
from scipy.special import softmax
from PIL import Image, ImageOps
import torch, joblib, time

torch.set_num_threads(2)
TASKS={'make':('make_recognition_images.csv','label_make'),
       'model':('recognition_images.csv','rec_label'),
       'color':('color_images.csv','label_color_std'),
       'body':('body_images.csv','label_body_type_std')}
RUN=Path(__file__).resolve().parents[1]/'models/improvement_20261024T223322912563/dual_encoder'
RUN.mkdir(parents=True,exist_ok=True)
BRAND=ROOT/'models/brand_final_20261022T074855/brand_model.pt'
BASE=ROOT/'models/20261021T090153837145_embeddings'
ACT=Path(os.environ['CAR_IMPROVEMENT_RUN'])/'activation_cache'

def cached_base(rows):
    idx=read_csv(BASE/'image_index.csv'); lookup={r['sha256']:i for i,r in enumerate(idx)}
    e=np.load(BASE/'embeddings.npy',mmap_mode='r'); c=np.load(BASE/'color_features.npy',mmap_mode='r')
    pos=[lookup[r['sha256']] for r in rows]
    return np.asarray(e[pos],dtype=np.float32),np.asarray(c[pos],dtype=np.float32)

def brand_cache(rows,flip=0):
    index=read_csv(ACT/'index.csv'); lookup={r['sha256']:i for i,r in enumerate(index)}
    ids=np.asarray([lookup[r['sha256']] for r in rows],dtype=np.int64)
    arr=np.load(ACT/f'stage2_{flip}.npy',mmap_mode='r')
    obj=torch.load(BRAND,map_location='cpu',weights_only=True)
    classes=obj['classes']; net=BrandNet(len(classes));net.load_state_dict(obj['state_dict']);net.eval()
    out=[];b=net.backbone
    with torch.inference_mode():
        for st in range(0,len(ids),64):
            z=torch.from_numpy(np.array(arr[ids[st:st+64]],copy=True))
            z=b.layer4(b.layer3(z));z=torch.nn.functional.normalize(b.avgpool(z).flatten(1),dim=1)
            out.append(z.numpy().astype(np.float32))
    return np.concatenate(out),classes

def image_features(rows,include_color=True):
    base,col=cached_base(rows);brand,brand_classes=brand_cache(rows,0)
    x=np.column_stack([base,brand,col]) if include_color else np.column_stack([base,brand])
    return x.astype(np.float32),brand_classes

def sample_weights(rows):
    counts=Counter(r['group_id'] for r in rows)
    w=np.asarray([1/counts[r['group_id']] for r in rows],dtype=np.float64);return w/w.mean()

def probabilities(model,scaler,x,classes,tau=0.):
    logits=model.decision_function(scaler.transform(x))
    if logits.ndim==1: logits=np.column_stack([-logits/2,logits/2])
    prior=np.asarray(model.class_prior_,dtype=np.float64)
    logits=logits-tau*np.log(np.maximum(prior,1e-12))[None,:]
    return softmax(logits,axis=1)

def metrics(y,p,classes):
    pred=np.asarray(classes)[np.argmax(p,axis=1)]
    return {'n':len(y),'accuracy':float(accuracy_score(y,pred)),
            'macro_F1':float(f1_score(y,pred,labels=classes,average='macro',zero_division=0))}

def fit_task(task):
    filename,target=TASKS[task]; tr=compact_rows(task,'train');va=compact_rows(task,'val')
    x,_=image_features(tr,task in ('make','color')); xv,_=image_features(va,task in ('make','color'))
    y=np.asarray([r[target] for r in tr]);yv=np.asarray([r[target] for r in va]);classes=sorted(set(y))
    if set(yv)-set(classes):raise ValueError(f'فئات val غير موجودة في train: {task}')
    weights=sample_weights(tr);trials=[];best=None
    for c in [.01,.1,1.,10.]:
        for balance in [None,'balanced']:
            scaler=StandardScaler().fit(x)
            model=LogisticRegression(C=c,class_weight=balance,max_iter=700,solver='lbfgs').fit(
                scaler.transform(x),y,sample_weight=weights)
            model.class_prior_=np.asarray([np.mean(y==cl) for cl in classes],dtype=np.float64)
            if list(model.classes_)!=classes:raise ValueError('تغير ترتيب الفئات')
            raw=probabilities(model,scaler,xv,classes,0.)
            for tau in [0.,.25,.5]:
                p=probabilities(model,scaler,xv,classes,tau)
                m=metrics(yv,p,classes)
                trial={'C':c,'class_weight':balance,'log_prior_tau':tau,**m}
                trials.append(trial)
                if best is None or m['macro_F1']>best['metrics']['macro_F1']:
                    best={'scaler':scaler,'model':model,'metrics':m,'tau':tau,'trial':trial,'probs':p}
    dest=RUN/task;dest.mkdir(exist_ok=True)
    joblib.dump({'model':best['model'],'scaler':best['scaler'],'classes':classes,'tau':best['tau'],
                 'target':target,'feature_schema':['resnet18_imagenet_512','brandnet_resnet18_512']+(['color_124'] if task in ('make','color') else []),
                 'fit_split':'train','encoder_fit_split':'train','seed':42},dest/'head.joblib',compress=3)
    save_json(dest/'validation.json',{'selected':best['trial'],'trials':trials,
        'listing_metric':listing_metrics(va,best['probs'],target,classes),'test_used':False,
        'feature_rows_train':len(tr),'feature_rows_val':len(va),'feature_dimensions':int(x.shape[1])})
    np.save(dest/'val_probabilities.npy',best['probs'])
    class_report(task,'val',va,best['probs'],classes)
    return {'classes':classes,'model':best['model'],'scaler':best['scaler'],'tau':best['tau'],
            'probs_val':best['probs'],'rows_val':va,'target':target,'metrics':best['metrics']}

def listing_metrics(rows,p,target,classes):
    grouped=defaultdict(list)
    for i,r in enumerate(rows):grouped[r['listing_uid']].append(i)
    y=[];q=[]
    for lid,ids in grouped.items():
        labels={rows[i][target] for i in ids}
        if len(labels)!=1:raise ValueError('تعارض وسوم داخل listing')
        y.append(labels.pop());q.append(p[ids].mean(axis=0))
    return metrics(np.asarray(y),np.asarray(q),classes)

def fit_semantic_fusions(fitted):
    # تعلم مصفوفة انتقال body|model من train فقط؛ val يحدد نسبة المزج.
    trb=compact_rows('body','train');trm=compact_rows('model','train')
    tr_by={r['image_uid']:r for r in trm}; body_classes=sorted({r['label_body_type_std'] for r in trb})
    model_classes=fitted['model']['classes']; counts=np.ones((len(model_classes),len(body_classes)),dtype=np.float64)
    for r in trb:
        mr=tr_by.get(r['image_uid'])
        if mr and mr['rec_label'] in model_classes and r['label_body_type_std'] in body_classes:
            counts[model_classes.index(mr['rec_label']),body_classes.index(r['label_body_type_std'])]+=1
    counts/=counts.sum(1,keepdims=True)
    body=fitted['body']; model=fitted['model']; rows=body['rows_val']; mrows=model['rows_val']
    bysha={r['sha256']:i for i,r in enumerate(mrows)}
    pm=np.zeros((len(rows),len(model_classes)),dtype=np.float64)
    for i,r in enumerate(rows):
        if r['sha256'] in bysha:pm[i]=model['probs_val'][bysha[r['sha256']]]
    trans=np.zeros((len(pm),len(body_classes)))
    for i,r in enumerate(rows):
        if r['sha256'] in bysha:
            for j,cl in enumerate(model_classes):trans[i]+=pm[i,j]*counts[j]
        else:
            trans[i]=body['probs_val'][i]
    direct=body['probs_val']; y=np.asarray([r['label_body_type_std'] for r in rows])
    best=(metrics(y,direct,body_classes)['macro_F1'],0.,direct)
    for w in [0.25,.5,.75]:
        p=(1-w)*direct+w*trans;p/=p.sum(1,keepdims=True)
        m=metrics(y,p,body_classes)
        if m['macro_F1']>best[0]:best=(m['macro_F1'],w,p)
    dest=RUN/'body';np.save(dest/'val_semantic_fusion.npy',best[2])
    save_json(dest/'semantic_fusion.json',{'transition_source':'train only','best_model_to_body_weight':best[1],
        'validation':metrics(y,best[2],body_classes),'test_used':False})

def val_ensemble():
    """مقارنة مزج محدود على val بين إعادة التدريب والمرجع المحلي الأعلى."""
    old_run=ROOT/'models/improvement_20261021T093545513846'
    report={}
    for task,(filename,target) in TASKS.items():
        new_rows=compact_rows(task,'val');new_art=joblib.load(RUN/task/'head.joblib')
        newp=np.load(RUN/task/'val_probabilities.npy');classes=new_art['classes']
        if task=='make':
            old_dir=ROOT/'models/brand_final_20261022T074855'
            oldp=np.load(old_dir/'val_probabilities.npy');oldids=read_csv(old_dir/'val_predictions.csv')
            oldobj=torch.load(old_dir/'brand_model.pt',map_location='cpu',weights_only=True);oldclasses=oldobj['classes']
        else:
            old_dir=old_run/task;oldp=np.load(old_dir/'val_probabilities.npy')
            oldids=read_csv(old_dir/'val_predictions.csv');oldclasses=load_json(old_dir/'selected.json')['classes']
        if list(map(str,classes))!=list(map(str,oldclasses)):
            report[task]={'status':'class_order_mismatch','new_classes':list(map(str,classes)),'old_classes':list(map(str,oldclasses))};continue
        lookup={r['image_uid']:i for i,r in enumerate(oldids)}
        if len(oldp)!=len(oldids) or any(r['image_uid'] not in lookup for r in new_rows):
            report[task]={'status':'row_alignment_mismatch'};continue
        aligned=np.asarray([oldp[lookup[r['image_uid']]] for r in new_rows])
        y=np.asarray([r[target] for r in new_rows]);candidates=[]
        for source,pnew in [('direct',newp),('body_semantic',np.load(RUN/'body/val_semantic_fusion.npy'))] if task=='body' else [('direct',newp)]:
            for w in [0.,.25,.5,.75,1.]:
                p=(1-w)*aligned+w*pnew
                m=metrics(y,p,classes);candidates.append({'candidate':source,'new_weight':w,**m})
        best=max(candidates,key=lambda r:r['macro_F1'])
        chosen=next((c for c in candidates if c==best),None)
        pnew=np.load(RUN/'body/val_semantic_fusion.npy') if task=='body' and best['candidate']=='body_semantic' else newp
        blended=(1-best['new_weight'])*aligned+best['new_weight']*pnew
        np.save(RUN/task/'val_best_blend_probabilities.npy',blended)
        report[task]={'reference':metrics(y,aligned,classes),'retrain_direct':metrics(y,newp,classes),
            'best_blend':best,'candidates':candidates,'selected_by':'val macro-F1 only'}
    save_json(RUN/'val_blend_comparison.json',report)
    return report

def test_fresh():
    lock=RUN/'test_selection_lock.json'
    if lock.exists():raise ValueError('اختيارات test مقفلة مسبقاً')
    # قفل كل أوزان الرؤوس قبل قراءة أي صف test.
    hashes={t:sha256(RUN/t/'head.joblib') for t in TASKS}
    save_json(lock,{'locked_at':now(),'head_sha256':hashes,'test_evaluations':0,'selection_split':'val'})
    tr=[]
    for task,(filename,target) in TASKS.items():tr.extend(compact_rows(task,'test'))
    unique={r['sha256']:r for r in tr};rows=list(unique.values())
    base,cols=cached_base(rows);brand,brand_classes=brand_cache_test(rows)
    # test rows are computed once into one immutable feature table for the four heads.
    results={};preds={}
    for task,(filename,target) in TASKS.items():
        taskrows=compact_rows(task,'test');idx={r['sha256']:i for i,r in enumerate(rows)}
        positions=np.asarray([idx[r['sha256']] for r in taskrows]);usecolor=task in ('make','color')
        x=np.column_stack([base[positions],brand[positions],cols[positions]]) if usecolor else np.column_stack([base[positions],brand[positions]])
        art=joblib.load(RUN/task/'head.joblib');p=probabilities(art['model'],art['scaler'],x,art['classes'],art['tau'])
        y=np.asarray([r[target] for r in taskrows]);m=metrics(y,p,art['classes']);results[task]=m;preds[task]=(taskrows,p,art['classes'])
        save_json(RUN/task/'test_once.json',{'state':'completed','n':len(taskrows),'evaluation_count':1,'metrics':m})
        np.save(RUN/task/'test_probabilities.npy',p)
    save_json(RUN/'test_results.json',{'results':results,'test_event_count_per_task':1,'test_historical_exposure':True})
    return preds,results

def brand_cache_test(rows):
    # استدلال مباشر بعد القفل على صور test فقط؛ لا يدخل الاختيار.
    obj=torch.load(BRAND,map_location='cpu',weights_only=True);classes=obj['classes'];net=BrandNet(len(classes));net.load_state_dict(obj['state_dict']);net.eval()
    from torchvision.models import ResNet18_Weights
    tf=ResNet18_Weights.IMAGENET1K_V1.transforms();out=[];b=net.backbone
    with torch.inference_mode():
        for st in range(0,len(rows),16):
            xs=[]
            for r in rows[st:st+16]:
                with Image.open(ROOT/'الملفات_المهمة'/r['relative_path']) as im:xs.append(tf(ImageOps.exif_transpose(im).convert('RGB')))
            z=b(torch.stack(xs));z=torch.nn.functional.normalize(z,dim=1);out.append(z.numpy().astype(np.float32))
    return np.concatenate(out),classes

def main():
    guard(); fitted={}
    # اختيار المرشحين على val فقط؛ لا تستدعى test قبل قفل جميع الرؤوس.
    for task in ['model','make','color','body']:fitted[task]=fit_task(task)
    fit_semantic_fusions(fitted)
    val_ensemble()
    lock=RUN/'selection_lock.json'
    save_json(lock,{'locked_at':now(),'val_only':True,'test_used_for_selection':False,
        'sha256':{t:sha256(RUN/t/'head.joblib') for t in TASKS}})
    save_json(RUN/'protocol.json',{'created_at':now(),'data_version':'day21-6-user-issues-reviewed',
        'split_fingerprint':load_json(PROCESSED/'training_manifest.json')['split_fingerprint'],
        'encoders':'frozen ImageNet ResNet18 + BrandNet trained on train only',
        'heads':'fresh LogisticRegression fit on train only; inverse group image-count weights',
        'pretest':'val candidate selection only','test':'one forward pass per task after lock',
        'test_historical_exposure':True})
    print('DUAL_ENCODER_VALIDATION_COMPLETE',flush=True)

if __name__=='__main__':main()
