"""تحسين بصري بإبطال تجربة لا تحسن macro-F1 للصورة على val."""
from improve_common import *
from ml_visual import embedding_data,task_features,probabilities,color_features,select_rejection_threshold
import argparse,time,copy,io
import joblib
from PIL import Image,ImageOps
import torch
from torch import nn
from torchvision.models import resnet18,ResNet18_Weights
from scipy.special import softmax
from scipy.optimize import minimize_scalar
torch.set_num_threads(4)


def base_data():
    cache,cm,lookup,e,c=embedding_data()
    return cache,lookup,e,c


def baseline_probs(task,rows):
    a=joblib.load(original(task)/'model.joblib')
    if rows and 'external_path' in rows[0]:
        from ml_visual import load_encoder,image_embedding
        encoder,tf=load_encoder(original('embeddings')/'encoder_weights.pth');features=[];colors=[]
        for r in rows:
            f,col=image_embedding(r['external_path'],encoder,tf);features.append(f);colors.append(col)
        return probabilities(a['model'],a['scaler'],task_features(np.asarray(features),np.asarray(colors),a['mode']),a['temperature']),a['classes']
    cache,lookup,e,c=base_data()
    pos=[lookup[r['sha256']] for r in rows]
    return probabilities(a['model'],a['scaler'],task_features(e[pos],c[pos],a['mode']),a['temperature']),a['classes']


def route_probs(pm,pc,model_classes,make_classes,threshold,method='sum',mask=None):
    derived=np.zeros_like(pm)
    if method=='hard':
        makers=[model_classes[i].split('|')[0] for i in pc.argmax(1)]
        for i,mk in enumerate(makers):derived[i,make_classes.index(mk)]=1.
    else:
        for j,label in enumerate(model_classes):derived[:,make_classes.index(label.split('|')[0])]+=pc[:,j]
    use=pc.max(1)>=threshold if mask is None else np.asarray(mask)
    return np.where(use[:,None],derived,pm)


def make_compare(improved=False):
    guard();rows=compact_rows('make','val');y=[r['label_make'] for r in rows]
    pm,classes=baseline_probs('make',rows)
    if improved:
        pc,model_classes=predict_selected('model',rows)
    else:pc,model_classes=baseline_probs('model',rows)
    before=score(y,pm,classes);best=before['macro_F1'];selected={'kind':'independent','model_source':'original'};bestp=pm
    path=session()/'make/selected.json'
    if improved and path.exists():
        prior=load_json(path);bestp=make_prediction(rows,prior)[0];best=score(y,bestp,classes)['macro_F1'];selected=prior
    for method in ['hard','sum']:
        for threshold in [0.,.2,.4,.6,.8,.95,1.01]:
            p=route_probs(pm,pc,model_classes,classes,threshold,method)
            m=score(y,p,classes);keep=m['macro_F1']>best+1e-8
            experiment('make',f'{"new" if improved else "old"}_model_{method}_{threshold}',
                'تغيير مصدر تنبؤ الماركة أو عتبة الرجوع؛ لا استخدام لهوية الموديل الحقيقية',best,m['macro_F1'],keep,m)
            if keep:
                best=m['macro_F1'];bestp=p;selected={'kind':'route','threshold':threshold,'method':method,'model_source':'improved' if improved else 'original'}
    oracle=route_probs(pm,pc,model_classes,classes,0,mask=[r['usable_recognition']=='True' for r in rows])
    save_json(session()/'make'/('oracle_improved_val.json' if improved else 'oracle_initial_val.json'),
        {'metrics':score(y,oracle,classes),'warning_ar':'مسار تحليلي يعتمد أهلية معلومة من الجدول، وليس توجيه صورة جديدة مجهولة؛ لا يعتمد في نشر المصنف'})
    selected.update(classes=classes,val_macro_F1=best)
    # عتبة قبول منفصلة من val باستعمال درجة المصدر الفعلية، لا قيمة hard=1.
    _,_,confidence=make_prediction(rows,selected,True)
    decision=np.full_like(bestp,0.)
    for i,j in enumerate(bestp.argmax(1)):
        decision[i]=(1-confidence[i])/(len(classes)-1);decision[i,j]=confidence[i]
    selected['rejection']=select_rejection_threshold(np.asarray(y),decision,classes)
    save_json(path,selected);np.save(session()/'make/val_probabilities.npy',bestp)
    class_report('make','val',rows,bestp,classes,np.asarray(classes)[pm.argmax(1)])
    log(session(),'اكتملت مقارنة الماركة بالتحقق فقط: '+str(selected))


def load_base():
    m=resnet18(weights=None);cache=original('embeddings')
    m.load_state_dict(torch.load(cache/'encoder_weights.pth',map_location='cpu',weights_only=True));return m.eval()


def build_cache():
    guard();s=session();folder=s/'activation_cache';folder.mkdir(exist_ok=False)
    unique={}
    for task in TASK_INFO:
        for r in compact_rows(task):
            if r['split']!='test':unique[r['sha256']]=r
    rows=[unique[k] for k in sorted(unique)];save_csv(folder/'index.csv',rows)
    model=load_base();tf=ResNet18_Weights.IMAGENET1K_V1.transforms()
    arrays={}
    for stage,shape in [(2,(128,28,28)),(3,(256,14,14))]:
        for flip in [0,1]:arrays[(stage,flip)]=np.lib.format.open_memmap(folder/f'stage{stage}_{flip}.npy',mode='w+',dtype='float32',shape=(len(rows),*shape))
    start=time.perf_counter()
    for first in range(0,len(rows),32):
        batch=rows[first:first+32];xs=[]
        for r in batch:
            p=ROOT/'الملفات_المهمة'/r['relative_path']
            if sha256(p)!=r['sha256']:raise ValueError('تغيرت صورة فعلية')
            with Image.open(p) as im:xs.append(tf(ImageOps.exif_transpose(im).convert('RGB')))
        x=torch.stack(xs);end=first+len(batch)
        with torch.inference_mode():
            for flip in [0,1]:
                z=x.flip(-1) if flip else x
                z=model.maxpool(model.relu(model.bn1(model.conv1(z))))
                z=model.layer2(model.layer1(z));arrays[(2,flip)][first:end]=z.numpy()
                z=model.layer3(z);arrays[(3,flip)][first:end]=z.numpy()
        if first%512==0 or end==len(rows):
            for a in arrays.values():a.flush()
            save_json(folder/'progress.json',{'done':end,'total':len(rows),'seconds':time.perf_counter()-start})
            print('ACTIVATIONS',end,'/',len(rows),round(time.perf_counter()-start,1),flush=True)
    save_json(folder/'complete.json',{'completed_at':now(),'n':len(rows),'test_images_in_cache':False,
        'views':'original and actual horizontal input flip; float32 activations, original images unchanged'})


class Tail(nn.Module):
    def __init__(self,task):
        super().__init__();a=joblib.load(original(task)/'model.joblib');self.backbone=load_base();self.backbone.fc=nn.Identity()
        self.mode=a['mode'];self.head=nn.Linear(512+(124 if self.mode=='combined' else 0),len(a['classes']))
        self.classes=a['classes'];coef=a['model'].coef_;intercept=a['model'].intercept_;sc=a['scaler']
        with torch.no_grad():
            self.head.weight.copy_(torch.tensor(coef/sc.scale_/a['temperature'],dtype=torch.float32))
            self.head.bias.copy_(torch.tensor((intercept-(coef*sc.mean_/sc.scale_).sum(1))/a['temperature'],dtype=torch.float32))
    def forward(self,x,extra=None,stage=0,return_features=False):
        b=self.backbone
        if stage==0:x=b.maxpool(b.relu(b.bn1(b.conv1(x))));x=b.layer2(b.layer1(x))
        if stage in (0,2):x=b.layer3(x)
        x=b.layer4(x);x=b.avgpool(x).flatten(1);x=nn.functional.normalize(x,dim=1)
        features=x
        if self.mode=='combined':x=torch.cat([x,extra],1)
        logits=self.head(x)
        return (logits,features) if return_features else logits


def activation_inputs(rows,stage,flip=0):
    folder=session()/'activation_cache';index=read_csv(folder/'index.csv');lookup={r['sha256']:i for i,r in enumerate(index)}
    array=np.load(folder/f'stage{stage}_{flip}.npy',mmap_mode='r')
    return array,np.asarray([lookup[r['sha256']] for r in rows])


def extra_inputs(rows):
    _,lookup,e,c=base_data();return np.asarray(c[[lookup[r['sha256']] for r in rows]])


def tail_val(model,rows,tta=False):
    array,ids=activation_inputs(rows,2,0);flip,_=activation_inputs(rows,2,1);extra=extra_inputs(rows);out=[]
    model.eval()
    with torch.inference_mode():
        for first in range(0,len(rows),64):
            selected=ids[first:first+64];col=torch.tensor(extra[first:first+64])
            p=model(torch.from_numpy(np.array(array[selected])),col,2).softmax(1)
            if tta:p=(p+model(torch.from_numpy(np.array(flip[selected])),col,2).softmax(1))/2
            out.append(p.numpy())
    return np.concatenate(out)


def save_tail(path,model,task,tta=False):
    buffer=io.BytesIO()
    torch.save({'state_dict':model.state_dict(),'classes':model.classes,'mode':model.mode,'task':task,'tta':tta,
                'input_sha256':load_json(session()/'protocol.json')['input_files'],'split_fingerprint':load_json(session()/'protocol.json')['split_fingerprint']},buffer)
    save_bytes(path,buffer.getvalue())


def fit_task(task):
    guard();folder=session()/task;tr=compact_rows(task,'train');va=compact_rows(task,'val');target=TASK_INFO[task][1]
    old,classes=baseline_probs(task,va);y=[r[target] for r in va];bestscore=score(y,old,classes)['macro_F1'];bestp=old
    selected={'kind':'original','task':task,'val_macro_F1':bestscore,'classes':classes}
    class_report(task,'val_before',va,old,classes)
    torch.manual_seed(42);np.random.seed(42);model=Tail(task);base_state=copy.deepcopy(model.state_dict())
    current={'depth':4,'weighted':False,'smoothing':0.,'augmentation':False,'lr':1e-5}
    trials=[('layer4',{'depth':4},3),('class_weights',{'weighted':True},2),('label_smoothing',{'smoothing':.05},2),
            ('horizontal_augmentation',{'augmentation':True},2),('lower_learning_rate',{'lr':3e-6},2),('layer3_and4',{'depth':3},2)]
    cmap={c:i for i,c in enumerate(classes)};yt=np.asarray([cmap[r[target]] for r in tr]);counts=np.bincount(yt,minlength=len(classes))
    extras=extra_inputs(tr);rng=np.random.default_rng(42)
    for name,change,epochs in trials:
        guard();rng=np.random.default_rng(42);torch.manual_seed(42);cfg={**current,**change};model.load_state_dict(base_state)
        for p in model.parameters():p.requires_grad=False
        blocks=[model.backbone.layer4]+([model.backbone.layer3] if cfg['depth']==3 else [])
        for block in blocks+[model.head]:
            for p in block.parameters():p.requires_grad=True
        params=[{'params':[p for b in blocks for p in b.parameters()],'lr':cfg['lr']},
                {'params':model.head.parameters(),'lr':cfg['lr']*5}]
        opt=torch.optim.AdamW(params,weight_decay=.01)
        weights=torch.tensor(np.sqrt(counts.sum()/np.maximum(counts,1)),dtype=torch.float32) if cfg['weighted'] else None
        if weights is not None:weights/=weights.mean()
        loss_fn=nn.CrossEntropyLoss(weight=weights,label_smoothing=cfg['smoothing'])
        stage=3 if cfg['depth']==4 else 2;array,ids=activation_inputs(tr,stage,0);flip,_=activation_inputs(tr,stage,1)
        before=bestscore;besttrial=-1.;trialstate=None;bad=0;start=time.perf_counter()
        for epoch in range(epochs):
            model.train()
            # لا تغير إحصاءات BN أثناء تدريب الذيل؛ prefix المخبأ ثابت تمامًا.
            for mod in model.modules():
                if isinstance(mod,nn.BatchNorm2d):mod.eval()
            order=rng.permutation(len(tr));losses=[]
            for first in range(0,len(order),32):
                chosen=order[first:first+32];pos=ids[chosen];x=np.array(array[pos])
                if cfg['augmentation']:
                    mask=rng.random(len(chosen))<.5
                    if mask.any():x[mask]=flip[pos[mask]]
                opt.zero_grad(set_to_none=True)
                logits=model(torch.from_numpy(x),torch.tensor(extras[chosen]),stage)
                loss=loss_fn(logits,torch.tensor(yt[chosen]));loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.);opt.step();losses.append(float(loss.detach()))
                if first%1024==0:print(task,name,'epoch',epoch+1,'images',min(first+32,len(tr)),'/',len(tr),flush=True)
            p=tail_val(model,va);m=score(y,p,classes)
            save_json(folder/f'{name}_epoch{epoch+1}.json',{'val':m,'loss':float(np.mean(losses)),'config':cfg,'seconds':time.perf_counter()-start})
            print(task,name,'EPOCH',epoch+1,'VAL',m,flush=True)
            if m['macro_F1']>besttrial:
                besttrial=m['macro_F1'];trialstate=copy.deepcopy(model.state_dict());bad=0
            else:bad+=1
            if bad>=1:break
        keep=besttrial>bestscore+1e-8
        experiment(task,name,json.dumps(change),bestscore,besttrial,keep,{'config':cfg,'epochs_completed':epoch+1,'seconds':time.perf_counter()-start})
        if keep:
            bestscore=besttrial;base_state=trialstate;current=cfg;model.load_state_dict(base_state)
            bestp=tail_val(model,va);save_tail(folder/'best_tail.pt',model,task)
            selected={'kind':'tail','task':task,'checkpoint':str((folder/'best_tail.pt').relative_to(session())),
                'tta':False,'classes':classes,'val_macro_F1':bestscore,'config':cfg}
        else:model.load_state_dict(base_state)
        save_json(folder/'selected.json',selected)
        np.save(folder/'val_probabilities.npy',bestp)
    model.load_state_dict(base_state);p=tail_val(model,va,tta=True);m=score(y,p,classes);keep=m['macro_F1']>bestscore+1e-8
    experiment(task,'horizontal_TTA','انعكاس أفقي عند الاستدلال فقط',bestscore,m['macro_F1'],keep,m)
    if keep:
        bestp=p;bestscore=m['macro_F1'];selected.update(kind='tail',checkpoint=str((folder/'best_tail.pt').relative_to(session())),tta=True,val_macro_F1=bestscore)
        save_tail(folder/'best_tail.pt',model,task,True)
    selected['rejection']=select_rejection_threshold(np.asarray(y),bestp,classes)
    save_json(folder/'selected.json',selected);np.save(folder/'val_probabilities.npy',bestp)
    class_report(task,'val',va,bestp,classes,np.asarray(classes)[old.argmax(1)])
    guard()


def load_tail(task,config):
    model=Tail(task);saved=torch.load(session()/config['checkpoint'],map_location='cpu',weights_only=False)
    model.load_state_dict(saved['state_dict']);return model.eval()


def predict_selected(task,rows,return_features=False):
    selected=load_json(session()/task/'selected.json')
    if selected['kind']=='original':return baseline_probs(task,rows)
    model=load_tail(task,selected)
    # cached prefix only contains train/val; test/external use real images in full forward.
    if not return_features and all(r.get('split') in ('train','val') and 'external_path' not in r for r in rows):
        return tail_val(model,rows,selected.get('tta',False)),selected['classes']
    tf=ResNet18_Weights.IMAGENET1K_V1.transforms();out=[];feature_out=[]
    with torch.inference_mode():
        for first in range(0,len(rows),32):
            xs=[];colors=[]
            for r in rows[first:first+32]:
                path=Path(r['external_path']) if 'external_path' in r else ROOT/'الملفات_المهمة'/r['relative_path']
                with Image.open(path) as src:
                    im=ImageOps.exif_transpose(src).convert('RGB');xs.append(tf(im));colors.append(color_features(im))
            x=torch.stack(xs);extra=torch.tensor(np.asarray(colors));logits,features=model(x,extra,0,True);p=logits.softmax(1)
            feature_out.append(features.numpy())
            if selected.get('tta'):p=(p+model(x.flip(-1),extra,0).softmax(1))/2
            out.append(p.numpy())
            if rows[0].get('split')=='test' and (first%512==0 or first+32>=len(rows)):
                print('FINAL FORWARD',task,min(first+32,len(rows)),'/',len(rows),flush=True)
    return (np.concatenate(out),selected['classes'],np.concatenate(feature_out)) if return_features else (np.concatenate(out),selected['classes'])


def make_prediction(rows,cfg=None,return_confidence=False):
    cfg=cfg or load_json(session()/'make/selected.json')
    if cfg.get('independent_source')=='adapted_head':
        from improve_make_head import make_features,head_probs
        head=joblib.load(session()/'make/adapted_head.joblib');pm=head_probs(make_features(rows),head);classes=head['classes']
    else:pm,classes=baseline_probs('make',rows)
    if cfg['kind']=='independent':return (pm,classes,pm.max(1)) if return_confidence else (pm,classes)
    pc,labels=predict_selected('model',rows) if cfg['model_source']=='improved' else baseline_probs('model',rows)
    p=route_probs(pm,pc,labels,classes,cfg['threshold'],cfg['method'])
    conf=np.where(pc.max(1)>=cfg['threshold'],pc.max(1),pm.max(1))
    return (p,classes,conf) if return_confidence else (p,classes)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('task',choices=['make','make_update','cache','model','color','body']);a=ap.parse_args()
    if a.task=='make':make_compare()
    elif a.task=='make_update':make_compare(True)
    elif a.task=='cache':build_cache()
    else:fit_task(a.task)
