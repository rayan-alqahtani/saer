"""بحث بمرجع تدريب ثابت واستبعاد الإعلان والمجموعة والبصمة صراحة."""
from improve_common import *
from ml_visual import embedding_data
from ml_price import metrics
from improve_visual import load_tail
from PIL import Image,ImageOps
from torchvision.models import ResNet18_Weights
import torch
from torch import nn
import numpy as np
import argparse


def tuned_features(rows):
    cfg=load_json(session()/'model/selected.json');m=load_tail('model',cfg);b=m.backbone
    act=session()/'activation_cache'
    if all('external_path' in r for r in rows):lookup={};array=None
    else:
        lookup={r['sha256']:i for i,r in enumerate(read_csv(act/'index.csv'))}
        array=np.load(act/'stage2_0.npy',mmap_mode='r')
    tf=ResNet18_Weights.IMAGENET1K_V1.transforms();out=[]
    with torch.inference_mode():
        for first in range(0,len(rows),32):
            batch=rows[first:first+32];cached=all(r.get('sha256') in lookup and 'external_path' not in r for r in batch)
            if cached:x=torch.from_numpy(np.array(array[[lookup[r['sha256']] for r in batch]]))
            else:
                xs=[]
                for r in batch:
                    path=Path(r['external_path']) if 'external_path' in r else ROOT/'الملفات_المهمة'/r['relative_path']
                    with Image.open(path) as im:xs.append(tf(ImageOps.exif_transpose(im).convert('RGB')))
                x=torch.stack(xs);x=b.layer2(b.layer1(b.maxpool(b.relu(b.bn1(b.conv1(x))))))
            x=b.layer4(b.layer3(x));x=b.avgpool(x).flatten(1);out.append(nn.functional.normalize(x,dim=1).numpy())
    return np.concatenate(out)


def evaluate(split,mode=None,prefix=None):
    rows=compact_rows('search');train=[r for r in rows if r['split']=='train']
    first={}
    for r in rows:
        if r['split']==split and r['group_id'] not in first:first[r['group_id']]=r
    queries=list(first.values());prefix=prefix or split
    if mode is None:mode=load_json(session()/'search/selected.json',{}).get('kind','frozen_resnet_cosine')
    if mode=='finetuned_cosine':
        indexed=read_csv(session()/'search/finetuned_index.csv');lookup={r['sha256']:i for i,r in enumerate(indexed)}
        e=np.load(session()/'search/finetuned_features.npy',mmap_mode='r');ev=e[[lookup[r['sha256']] for r in train]]
        qvectors=e[[lookup[r['sha256']] for r in queries]] if split=='val' else tuned_features(queries)
    else:
        cache,cm,lookup,e,c=embedding_data();ev=e[[lookup[r['sha256']] for r in train]];qvectors=e[[lookup[r['sha256']] for r in queries]]
    classes=sorted({r['rec_label'] for r in train});hits1=[];hits5=[];pred=[];records=[];exclusions=[]
    for n,q in enumerate(queries):
        qv=qvectors[n];sim=ev@qv;order=np.argsort(-sim);seen=set();found=[];excluded=0
        for i in order:
            r=train[i]
            if r['listing_uid']==q['listing_uid'] or r['group_id']==q['group_id'] or r['sha256']==q['sha256']:
                excluded+=1;continue
            if r['group_id'] in seen:continue
            seen.add(r['group_id']);found.append(r)
            if len(found)==5:break
        if not found:raise ValueError('لم يبق معرض بحث صالح')
        hits1.append(float(found[0]['rec_label']==q['rec_label']));hits5.append(float(any(r['rec_label']==q['rec_label'] for r in found)))
        pred.append(found[0]['rec_label']);exclusions.append(excluded)
        records.append({'query_uid':q['image_uid'],'listing_uid':q['listing_uid'],'group_id':q['group_id'],'actual':q['rec_label'],
            'top1_label':found[0]['rec_label'],'top1_hit':hits1[-1],'top5_hit':hits5[-1],
            'retrieved_listing_uids':';'.join(r['listing_uid'] for r in found),'same_listing_returned':any(r['listing_uid']==q['listing_uid'] for r in found),
            'same_sha_returned':any(r['sha256']==q['sha256'] for r in found),'same_group_returned':any(r['group_id']==q['group_id'] for r in found)})
    save_csv(session()/'search'/f'{prefix}_raw_predictions.csv',records)
    weights=np.random.default_rng(42).multinomial(len(queries),np.full(len(queries),1/len(queries)),size=2000)
    ci={'top1':np.quantile(weights@np.asarray(hits1)/len(queries),[.025,.975]).tolist(),
        'top5':np.quantile(weights@np.asarray(hits5)/len(queries),[.025,.975]).tolist(),'bootstrap_unit':'group_id','repetitions':2000}
    per=classification_report([r['rec_label'] for r in queries],pred,labels=classes,output_dict=True,zero_division=0)
    save_csv(session()/'search'/f'{prefix}_per_class.csv',[{'class':cl,**per[cl]} for cl in classes])
    save_csv(session()/'search'/f'{prefix}_worst10.csv',[{'class':cl,**per[cl],'possible_reason_ar':'تشابه بصري أو نقص أمثلة؛ قياس الملاءمة بالموديل لا بهوية السيارة'} for cl in sorted([c for c in classes if per[c]['support']>0],key=lambda c:per[c]['f1-score'])[:10]])
    macro_ci,_,_=class_bootstrap(queries,[r['rec_label'] for r in queries],pred,classes)
    save_csv(session()/'search'/f'{prefix}_per_class.csv',[{'class':cl,**per[cl],
        **{f'{metric}_ci95_{side}':macro_ci['per_class'][cl][metric][index] for metric in ['precision','recall','F1'] for index,side in enumerate(['low','high'])}} for cl in classes])
    result={'n_query_groups':len(queries),'top1':float(np.mean(hits1)),'top5':float(np.mean(hits5)),
        'top1_macro_F1':float(f1_score([r['rec_label'] for r in queries],pred,labels=classes,average='macro',zero_division=0)),
        'ci95':ci,'macro_F1_ci95':macro_ci['macro_F1'],'encoder_kind':mode,'same_listing_or_sha_or_group_returned':any(r['same_listing_returned'] or r['same_sha_returned'] or r['same_group_returned'] for r in records),
        'gallery_split':'train','query_policy':'first image per group in fixed order','relevance':'same rec_label',
        'reference_top5':load_json(original('search')/'metrics.json')['results'][split]['model_hit_at_5'],
        'note_ar':'المعرض والسياسة ثابتان؛ المقارنة بين تمثيل ResNet18 الأصلي وتمثيله المضبوط على train. الاستبعادات صريحة.'}
    save_json(session()/'search'/f'{prefix}_metrics.json',result);save_csv(session()/'search'/f'{prefix}_predictions.csv',records)
    return result


def prepare():
    guard();m=evaluate('val',mode='frozen_resnet_cosine',prefix='val_before');selected={'kind':'frozen_resnet_cosine','gallery':'train','k':[1,5],
        'exclude':['listing_uid','sha256','group_id'],'baseline_encoder':str(original('embeddings').relative_to(ROOT))}
    experiment('search','exclude_self_explicitly','استبعاد الإعلان والبصمة والمجموعة عند الاستعلام',m['reference_top5'],m['top5'],True,{'validation':m,'policy_fixed_not_hyperparameter_selection':True})
    if load_json(session()/'model/selected.json')['kind']=='tail':
        rows=[r for r in compact_rows('search') if r['split']!='test'];save_csv(session()/'search/finetuned_index.csv',rows)
        print('SEARCH building fine-tuned train/val index',len(rows),flush=True)
        features=tuned_features(rows);np.save(session()/'search/finetuned_features.npy',features)
        candidate=evaluate('val',mode='finetuned_cosine',prefix='val_candidate');keep=candidate['top5']>m['top5']+1e-8
        experiment('search','finetuned_encoder','تغيير المشفر إلى الموديل المضبوط فقط؛ معيار val top5',m['top5'],candidate['top5'],keep,candidate)
        if keep:selected['kind']='finetuned_cosine'
    save_json(session()/'search/selected.json',selected);evaluate('val',mode=selected['kind'])


if __name__=='__main__':prepare()
