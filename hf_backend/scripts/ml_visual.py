"""تمثيلات مجمدة مشتركة ومصنفات مستقلة وبحث بصري على CPU."""
from ml_common import *
import argparse
import math
import shutil
import time
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np
import joblib
from PIL import Image, ImageOps, UnidentifiedImageError
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix, log_loss
from scipy.optimize import minimize_scalar
from scipy.special import softmax

TASKS={'make':('make_recognition_images.csv','label_make'),
       'model':('recognition_images.csv','rec_label'),
       'color':('color_images.csv','label_color_std'),
       'body':('body_images.csv','label_body_type_std')}


def color_features(im):
    """إحصاءات لونية للصورة كاملة والوسط؛ لا تدعي فصل السيارة عن الخلفية."""
    im=ImageOps.exif_transpose(im).convert('RGB')
    w,h=im.size
    crops=[im,im.crop((int(.18*w),int(.2*h),int(.82*w),int(.85*h)))]
    out=[]
    for crop in crops:
        a=np.asarray(crop.resize((96,72)).convert('HSV'),dtype=float)/255.
        # توزيع لوني لا يستخدم أي وسم أو معلومة إعلان.
        for c,bins in [(0,24),(1,16),(2,16)]:
            out.extend(np.histogram(a[:,:,c],bins=bins,range=(0,1))[0]/a.shape[0]/a.shape[1])
        rgb=np.asarray(crop.resize((96,72)),dtype=float)/255.
        out.extend(np.mean(rgb,axis=(0,1))); out.extend(np.std(rgb,axis=(0,1)))
    return np.asarray(out,dtype=np.float32)


def load_encoder(weights):
    import torch
    from torchvision.models import resnet18, ResNet18_Weights
    torch.set_num_threads(4)
    torch.manual_seed(SEED)
    model=resnet18(weights=None)
    model.load_state_dict(torch.load(weights,map_location='cpu',weights_only=True))
    model.fc=torch.nn.Identity()
    return model.eval(),ResNet18_Weights.IMAGENET1K_V1.transforms()


def image_embedding(photo, model, transform):
    import torch
    try:
        with Image.open(photo) as src:
            if src.width*src.height>40_000_000: raise ValueError('أبعاد الصورة تتجاوز الحد')
            im=ImageOps.exif_transpose(src).convert('RGB')
            color=color_features(im)
            tensor=transform(im)
    except (UnidentifiedImageError,OSError) as exc:
        raise ValueError('الصورة غير صالحة أو غير قابلة للقراءة') from exc
    with torch.inference_mode():
        x=model(tensor.unsqueeze(0)).numpy()[0]
    x=x/max(float(np.linalg.norm(x)),1e-12)
    return x.astype(np.float32),color


def embed():
    import torch
    folder,meta=new_run('embeddings')
    weights=Path.home()/'.cache/torch/hub/checkpoints/resnet18-f37072fd.pth'
    if not weights.exists() or not sha256(weights).startswith('f37072fd'): raise ValueError('أوزان محلية موثقة غير متاحة')
    shutil.copy2(weights,folder/'encoder_weights.pth')
    model,transform=load_encoder(folder/'encoder_weights.pth')
    all_rows=[]
    for name in [*(v[0] for v in TASKS.values()),'visual_search_images.csv']:
        rows=read_csv(PROCESSED/name); assert_split_integrity(rows,image_task=True); all_rows.extend(rows)
    unique={r['sha256']:r for r in all_rows}
    rows=[unique[k] for k in sorted(unique)]
    save_csv(folder/'image_index.csv',[{k:r[k] for k in ('sha256','image_uid','listing_uid','group_id','split','relative_path')} for r in rows])
    e=np.lib.format.open_memmap(folder/'embeddings.npy',mode='w+',dtype='float32',shape=(len(rows),512))
    c=np.lib.format.open_memmap(folder/'color_features.npy',mode='w+',dtype='float32',shape=(len(rows),124))
    meta.update(encoder='resnet18_imagenet1k_v1_frozen',weights_sha256=sha256(weights),
                weights_source='https://download.pytorch.org/models/resnet18-f37072fd.pth',
                weights_docs='https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html',
                preprocessing='torchvision IMAGENET1K_V1: resize shorter side 256, center crop 224, ImageNet normalization',
                fit_on_project_images=False,n_images=len(rows),embedding_dimensions=512,color_dimensions=124,
                batch_size=32,original_images_modified=False)
    save_json(folder/'metadata.json',meta)
    start=time.perf_counter(); batch_size=32
    log(folder,f'بدء تمثيلات {len(rows)} صورة فريدة على CPU؛ أوزان ImageNet مجمدة ومحفوظة محليًا')
    for start_idx in range(0,len(rows),batch_size):
        batch=rows[start_idx:start_idx+batch_size]; tensors=[]; colors=[]
        for r in batch:
            path=ROOT/r['relative_path']
            if sha256(path)!=r['sha256']: raise ValueError('تغيرت صورة منذ التدقيق')
            with Image.open(path) as src:
                im=ImageOps.exif_transpose(src).convert('RGB')
                tensors.append(transform(im)); colors.append(color_features(im))
        with torch.inference_mode():
            f=model(torch.stack(tensors)).numpy()
        f=f/np.maximum(np.linalg.norm(f,axis=1,keepdims=True),1e-12)
        end=start_idx+len(batch); e[start_idx:end]=f; c[start_idx:end]=colors
        if start_idx==0 or end==len(rows) or (start_idx//batch_size)%16==0:
            e.flush(); c.flush(); elapsed=time.perf_counter()-start
            save_json(folder/'progress.json',{'completed':end,'total':len(rows),'elapsed_seconds':elapsed,
                'estimated_remaining_seconds':elapsed/end*(len(rows)-end)})
            log(folder,f'التمثيلات {end}/{len(rows)}؛ {elapsed:.1f} ثانية')
    if not np.isfinite(e).all() or not np.isfinite(c).all(): raise ValueError('قيم غير صالحة في التمثيلات')
    meta.update(training_status='completed',completed_at=now(),elapsed_seconds=time.perf_counter()-start)
    save_json(folder/'metadata.json',meta); verified_inputs()
    save_json(ROOT/'models/embeddings_current.json',{'run_dir':str(folder.relative_to(ROOT)),**{k:meta[k] for k in ('dataset_version','dataset_fingerprint','split_fingerprint')}})
    print('EMBEDDINGS_RUN='+str(folder),flush=True)


def embedding_data():
    current=load_json(ROOT/'models/embeddings_current.json')
    identity=verified_inputs(); assert_artifact_compatible(current,identity)
    folder=ROOT/current['run_dir']; meta=load_json(folder/'metadata.json')
    assert_artifact_compatible(meta,identity)
    index=read_csv(folder/'image_index.csv')
    e=np.load(folder/'embeddings.npy',mmap_mode='r'); c=np.load(folder/'color_features.npy',mmap_mode='r')
    return folder,meta,{r['sha256']:i for i,r in enumerate(index)},e,c


def task_features(e,c,mode):
    if mode=='embeddings': return np.asarray(e)
    if mode=='color': return np.asarray(c)
    if mode=='combined': return np.column_stack([e,c])
    raise ValueError(mode)


def aggregate_groups(rows,probs,label_field):
    # group_id لعزل التسرب وقد يضم إعلانات مختلفة؛ نجمع صور الإعلان نفسه فقط.
    groups=defaultdict(list)
    for i,r in enumerate(rows): groups[r['listing_uid']].append(i)
    targets=[]; means=[]; ids=[]
    for group,indices in groups.items():
        labels={rows[i][label_field] for i in indices}
        if len(labels)!=1:
            raise ValueError('تعارض هدف الإعلان في المهمة: '+group)
        targets.append(labels.pop()); means.append(probs[indices].mean(axis=0)); ids.append(group)
    return np.asarray(targets),np.asarray(means),ids


def class_metrics(y,probs,classes):
    pred=np.asarray(classes)[np.argmax(probs,axis=1)]
    top=np.argsort(-probs,axis=1)[:,:min(3,len(classes))]
    return {'n':len(y),'accuracy':float(accuracy_score(y,pred)),
            'macro_F1':float(f1_score(y,pred,labels=classes,average='macro',zero_division=0)),
            'macro_recall':float(np.mean([np.mean(pred[np.asarray(y)==cl]==cl) if np.any(np.asarray(y)==cl) else 0 for cl in classes])),
            'top3_accuracy':float(np.mean([str(label) in np.asarray(classes)[idx] for label,idx in zip(y,top)])),
            'log_loss':float(log_loss(y,probs,labels=classes))}


def probabilities(model,scaler,x,temperature=1.):
    scores=model.decision_function(scaler.transform(x))
    if scores.ndim==1: scores=np.column_stack([-scores/2,scores/2])
    return softmax(scores/temperature,axis=1)


def select_rejection_threshold(y,p,classes,target_precision=.90):
    confidence=p.max(axis=1); pred=np.asarray(classes)[p.argmax(axis=1)]; best=None
    # حد عدم اليقين من التحقق فقط؛ لا يثبت كشف كل صورة خارج النطاق.
    for threshold in np.linspace(.3,.99,70):
        keep=confidence>=threshold; n=int(keep.sum())
        if n>=max(20,math.ceil(len(y)*.1)):
            precision=float(np.mean(pred[keep]==np.asarray(y)[keep]))
            if precision>=target_precision:
                best={'threshold':float(threshold),'validation_precision':precision,'validation_coverage':n/len(y),'n':n}; break
    return best or {'threshold':1.01,'validation_precision':None,'validation_coverage':0.,'n':0,
                    'reason_ar':'لم يدعم التحقق حد قبول تلقائي بدقة 90% وتغطية كافية؛ تعرض اقتراحات غير مؤكدة'}


def fit_classifier(task):
    cache,cache_meta,lookup,e,c=embedding_data()
    folder,meta=new_run(task)
    filename,target=TASKS[task]; rows=read_csv(PROCESSED/filename)
    assert_split_integrity(rows,image_task=True)
    indices={s:[i for i,r in enumerate(rows) if r['split']==s] for s in ('train','val','test')}
    positions=np.asarray([lookup[r['sha256']] for r in rows]); ee=e[positions]; cc=c[positions]
    tr,va,te=([rows[i] for i in indices[s]] for s in ('train','val','test'))
    yt,yv,ye=(np.asarray([r[target] for r in rr]) for rr in (tr,va,te))
    classes=sorted(set(yt))
    if set(yv)-set(classes) or set(ye)-set(classes): raise ValueError('فئة غير متعلمة في قسم التقييم')
    group_count=Counter(r['group_id'] for r in tr)
    sample_weight=np.asarray([1/group_count[r['group_id']] for r in tr]); sample_weight/=sample_weight.mean()
    modes=['embeddings','combined'] if task=='color' else ['embeddings']
    plans=[(mode,C,weight) for mode in modes for C in [.1,1.,10.] for weight in [None,'balanced']]
    save_json(folder/'plan.json',{'candidates':plans,'target':target,'selection':'validation_listing_macro_F1',
        'aggregation_unit':'listing_uid','group_role':'leakage isolation; targets may differ within a group',
        'test_used_for_selection':False,'fit_split':'train','sample_weight':'inverse_images_per_train_group',
        'classes':classes,'class_count':len(classes),'task_input_sha256':sha256(PROCESSED/filename),
        'encoder_run':str(cache.relative_to(ROOT)),'encoder_fit_on_project_images':False})
    log(folder,f'{task}: فئات={len(classes)} train={len(tr)} val={len(va)} test={len(te)}')
    start=time.perf_counter(); trials=[]; best=None
    for mode,C,weight in plans:
        x=task_features(ee,cc,mode); xt=x[indices['train']]; xv=x[indices['val']]
        scaler=StandardScaler().fit(xt)
        model=LogisticRegression(C=C,class_weight=weight,max_iter=600,tol=1e-4,solver='lbfgs')
        t=time.perf_counter(); model.fit(scaler.transform(xt),yt,sample_weight=sample_weight)
        p=probabilities(model,scaler,xv)
        gy,gp,_=aggregate_groups(va,p,target); m=class_metrics(gy,gp,classes)
        trial={'mode':mode,'C':C,'class_weight':weight,'validation_listing':m,'fit_seconds':time.perf_counter()-t,
               'iterations':int(model.n_iter_.max()),'converged':bool(model.n_iter_.max()<600)}
        trials.append(trial); save_json(folder/'selection.json',trials)
        log(folder,f'{task} {mode} C={C} weight={weight}: val listing macroF1={m["macro_F1"]:.3f} accuracy={m["accuracy"]:.3f}')
        if best is None or m['macro_F1']>best[0]['validation_listing']['macro_F1']:
            best=(trial,model,scaler)
    chosen,model,scaler=best; x=task_features(ee,cc,chosen['mode'])
    save_json(folder/'selection_lock.json',{'selected':chosen,'locked_at':now(),'test_not_evaluated_yet':True})
    # معايرة مقياس درجات المصنف على التحقق دون استخدام الاختبار.
    pv=probabilities(model,scaler,x[indices['val']]); gy,_,_=aggregate_groups(va,pv,target)
    scores=model.decision_function(scaler.transform(x[indices['val']]))
    if scores.ndim==1: scores=np.column_stack([-scores/2,scores/2])
    def temperature_loss(logt):
        pp=softmax(scores/np.exp(logt),axis=1); yy,gg,_=aggregate_groups(va,pp,target)
        return log_loss(yy,gg,labels=classes)
    fit=minimize_scalar(temperature_loss,bounds=(-2.3,2.3),method='bounded')
    temperature=float(np.exp(fit.x))
    pv=probabilities(model,scaler,x[indices['val']],temperature); gy,gp,_=aggregate_groups(va,pv,target)
    rejection=select_rejection_threshold(gy,gp,classes)
    image_rejection=select_rejection_threshold(yv,pv,classes)
    artifact={'model':model,'scaler':scaler,'temperature':temperature,'rejection':image_rejection,
              'listing_rejection':rejection,'classes':classes,
              'mode':chosen['mode'],'task':task,'target':target,'encoder_run':str(cache.relative_to(ROOT)),
              'metadata':meta}
    joblib.dump(artifact,folder/'model.joblib',compress=3)
    # الاختبار بعد تثبيت الاختيار، لا تدخل نتائجه في تحسين لاحق لهذا التشغيل.
    pp=probabilities(model,scaler,x[indices['test']],temperature)
    yy,gg,gids=aggregate_groups(te,pp,target)
    report=classification_report(yy,np.asarray(classes)[gg.argmax(axis=1)],labels=classes,output_dict=True,zero_division=0)
    per_class=[]
    for cl in classes:
        per_class.append({'class':cl,**report[cl],
            'train_groups':len({r['group_id'] for r in tr if r[target]==cl}),
            'val_groups':len({r['group_id'] for r in va if r[target]==cl}),
            'test_groups':len({r['group_id'] for r in te if r[target]==cl})})
    save_csv(folder/'test_per_class.csv',per_class)
    save_json(folder/'test_confusion_matrix.json',{'classes':classes,'matrix':confusion_matrix(yy,np.asarray(classes)[gg.argmax(axis=1)],labels=classes).tolist(),'unit':'listing_uid'})
    pred_rows=[]
    for group,label,probs in zip(gids,yy,gg):
        order=np.argsort(-probs)[:3]
        pred_rows.append({'listing_uid':group,'actual':label,'predicted':classes[order[0]],'score':float(probs[order[0]]),
            'top3':';'.join(classes[i] for i in order),'accepted':bool(probs[order[0]]>=rejection['threshold'])})
    save_csv(folder/'test_predictions.csv',pred_rows)
    baseline=Counter(r[target] for r in tr).most_common(1)[0][0]
    keep=gg.max(axis=1)>=rejection['threshold']
    result={'task':task,'class_count':len(classes),'classes':classes,'selected':chosen,
            'test_images':class_metrics(ye,pp,classes),'test_listings':class_metrics(yy,gg,classes),
            'dummy_majority_listing_accuracy':float(np.mean(yy==baseline)),
            'split_images':{s:len(indices[s]) for s in indices},
            'split_groups':{s:len({rows[i]['group_id'] for i in ix}) for s,ix in indices.items()},
            'temperature':temperature,'listing_rejection':rejection,'image_rejection':image_rejection,
            'test_selective':{'coverage':float(keep.mean()),'n':int(keep.sum()),
                'accuracy':float(np.mean(np.asarray(classes)[gg.argmax(axis=1)][keep]==yy[keep])) if keep.any() else None},
            'status':'experimental','elapsed_seconds':time.perf_counter()-start,
            'external_evaluation':'لم يختبر بعد',
            'limits_ar':['وسوم إعلانات ومصدر واحد؛ لا تعميم مثبت على صور الهاتف','قياس الصور المنفردة وقياس متوسط صور الإعلان منفصلان',
                         'حد الرفض مشتق من إعلانات التحقق ولا يثبت كشف الصور خارج النطاق','الموديلات المدعومة فقط ولا ادعاء تغطية جميع السيارات',
                         'اللون قد يتأثر بالخلفية والإضاءة؛ لم ينفذ فصل للسيارة']}
    save_json(folder/'metrics.json',result)
    meta.update(training_status='completed',completed_at=now(),status='experimental',class_count=len(classes),
                encoder_run=str(cache.relative_to(ROOT)),selected=chosen)
    save_json(folder/'metadata.json',meta); verified_inputs()
    save_json(ROOT/f'models/{task}_current.json',{'run_dir':str(folder.relative_to(ROOT)),**{k:meta[k] for k in ('dataset_version','dataset_fingerprint','split_fingerprint')}})
    log(folder,f'{task} اكتمل: test listing {result["test_listings"]}')


def search_vectors(q,e,rows,k=5,exclude_group=None,exclude_sha=None):
    scores=e@q; order=np.argsort(-scores); chosen=[]; seen=set()
    for i in order:
        r=rows[i]
        if r['group_id']==exclude_group or r['sha256']==exclude_sha or r['group_id'] in seen: continue
        seen.add(r['group_id']); chosen.append({**r,'similarity':float(scores[i])})
        if len(chosen)>=k: break
    return chosen


def build_search():
    cache,cm,lookup,e,c=embedding_data(); folder,meta=new_run('search')
    rows=read_csv(PROCESSED/'visual_search_images.csv'); assert_split_integrity(rows,image_task=True)
    vectors=np.asarray(e[[lookup[r['sha256']] for r in rows]])
    save_csv(folder/'image_index.csv',rows); np.save(folder/'embeddings.npy',vectors)
    meta.update(encoder_run=str(cache.relative_to(ROOT)),index_scope='train_only',query_splits=['val','test'],
                k=5,score_ar='تشابه كوني وليس احتمال صحة',external_evaluation='لم يختبر بعد')
    start=time.perf_counter(); results={}; examples=[]
    train_rows=[r for r in rows if r['split']=='train']; train_vectors=vectors[[i for i,r in enumerate(rows) if r['split']=='train']]
    train_labels=defaultdict(set)
    for r in train_rows: train_labels[r['rec_label']].add(r['group_id'])
    for split in ('val','test'):
        # صورة أولى ثابتة لكل مجموعة؛ لا اختيار الصورة الأفضل بحسب نتيجة البحث.
        first={}
        for i,r in enumerate(rows):
            if r['split']==split and r['group_id'] not in first: first[r['group_id']]=i
        hits=[]; precision=[]; recalls=[]; makes=[]; unsupported=0
        log(folder,f'تقييم البحث: {split}، {len(first)} استعلام مجموعة، الفهرس تدريب فقط')
        for n,i in enumerate(first.values()):
            r=rows[i]; found=search_vectors(vectors[i],train_vectors,train_rows,k=5,exclude_group=r['group_id'],exclude_sha=r['sha256'])
            available=len(train_labels[r['rec_label']])
            same=sum(f['rec_label']==r['rec_label'] for f in found)
            hit=float(same>0); hits.append(hit); precision.append(same/len(found))
            makes.append(float(any(f['label_make']==r['label_make'] for f in found)))
            if available: recalls.append(same/available)
            else: unsupported+=1
            if n<10:
                examples.append({'split':split,'query_image':r['relative_path'],'query_group':r['group_id'],'query_label':r['rec_label'],
                    'results':[{'image':f['relative_path'],'listing_uid':f['listing_uid'],'group_id':f['group_id'],'label':f['rec_label'],'score':f['similarity']} for f in found]})
        results[split]={'n_query_groups':len(first),'model_hit_at_5':float(np.mean(hits)),
            'make_hit_at_5':float(np.mean(makes)),'model_precision_at_5':float(np.mean(precision)),
            'model_recall_at_5_supported_queries':float(np.mean(recalls)) if recalls else None,
            'queries_without_same_model_in_train':unsupported,
            'query_policy':'first eligible image per group, fixed file order',
            'relevance':'same make|model label, not human judged instance similarity'}
    save_json(folder/'metrics.json',{'status':'experimental','results':results,'k':5,
        'index_train_images':len(train_rows),'index_train_groups':len({r['group_id'] for r in train_rows}),
        'elapsed_seconds':time.perf_counter()-start,'external_evaluation':'لم يختبر بعد',
        'limits_ar':['تشابه تمثيلات عامة مجمدة وليس هوية سيارة مثبتة','ملاءمة البحث تقاس باتفاق الماركة والموديل لا حكم بشري لكل زوج','لا صور استعلام من نفس مجموعة الفهرس أو البصمة']})
    save_json(folder/'query_examples.json',examples)
    meta.update(training_status='completed',completed_at=now(),status='experimental')
    save_json(folder/'metadata.json',meta); verified_inputs()
    save_json(ROOT/'models/search_current.json',{'run_dir':str(folder.relative_to(ROOT)),**{k:meta[k] for k in ('dataset_version','dataset_fingerprint','split_fingerprint')}})
    log(folder,'اكتمل فهرس البحث وتقييمه: '+str(results))


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('command',choices=['embed','make','model','color','body','search'])
    args=parser.parse_args()
    if args.command=='embed': embed()
    elif args.command=='search': build_search()
    else: fit_classifier(args.command)
