"""تشغيل نسخة الجولة البصرية على صور المستخدم؛ المرشح المقبول أو المرجع المحفوظ."""
import argparse,json,sys
from pathlib import Path
import numpy as np
import joblib
import visual_fusion_round as train

def extract(paths):
    import torch
    from PIL import Image,ImageOps
    from torchvision.transforms.functional import to_tensor
    from torchvision.models import resnet18,ResNet18_Weights
    from ml_visual import color_features
    d,det,tf=train.models();res=resnet18(weights=None)
    res.load_state_dict(torch.load(train.HUB/'checkpoints/resnet18-f37072fd.pth',map_location='cpu',weights_only=True));res.fc=torch.nn.Identity();res.eval()
    out=[]
    with torch.inference_mode():
        for path in paths:
            path=Path(path)
            if not path.is_file() or path.stat().st_size>15*1024*1024:raise ValueError('صورة غير موجودة أو أكبر من 15 ميجابايت')
            with Image.open(path) as src:
                if src.width*src.height>40000000:raise ValueError('أبعاد الصورة كبيرة جدًا')
                im=ImageOps.exif_transpose(src).convert('RGB')
            crop,box,score,found=train.crop_image(im,det([to_tensor(im)])[0])
            base=torch.nn.functional.normalize(res(ResNet18_Weights.IMAGENET1K_V1.transforms()(im)[None]),dim=1).numpy()[0]
            pair=d(torch.stack([tf(im),tf(crop)])).numpy()
            out.append({'full':pair[0],'crop':pair[1],'base':base,'original_color':color_features(im),'crop_color':train.color_desc(crop),'box':box,'detected':found,'detector_score':score})
    return out

def features(f,kind,task):
    if kind=='color_only':return np.concatenate([f['original_color'],f['crop_color']])[None]
    parts={'dino':[f['full']],'fusion':[f['full'],f['base']],'crop':[f['crop'],f['base']],'both':[f['full'],f['crop'],f['base']]}[kind]
    if task=='color':parts=parts+[f['original_color']]+([f['crop_color']] if kind in ('crop','both') else [])
    return np.concatenate(parts)[None]

def direct(task,f):
    a=joblib.load(train.RUN/task/'head.joblib')
    return a['model'].predict_proba(a['scaler'].transform(features(f,a['kind'],task))),a['classes']

def predict(task,path,f=None,force_candidate=False):
    from rescue_six_predict import predict_image
    r=train.load_json(train.RUN/task/'result.json')
    if not force_candidate and not r['accepted']:return predict_image(task,path)
    if f is None:f=extract([path])[0]
    p,cs=direct(task,f);name=r['selected_val_option']
    if name=='half_reference':
        ref,rc=predict_image(task,path);assert list(cs)==list(rc);p=(p+ref)/2
    elif name=='hierarchy_fixed':
        pm,mc=direct('model',f);mat=np.load(train.RUN/'make/model_to_make.npy')
        if pm.max()>=.7:p=.5*p+.5*(pm@mat)
    return p,cs

def search(path,f=None):
    # Latest user-approved development search candidate: frozen DINOv2 + PCA256 whitening.
    candidate=train.RUN/'search_candidate_pca256'
    if (candidate/'decision.json').is_file():
        meta=train.load_json(candidate/'decision.json')
        if f is None:f=extract([path])[0]
        q=train.norm(joblib.load(candidate/'pca.joblib').transform(train.norm(f['full'][None])))
        gallery=train.read_csv(candidate/'gallery.csv');g=np.load(candidate/'gallery.npy',mmap_mode='r')
        if len(g)!=len(gallery) or g.shape[1]!=q.shape[1]:raise ValueError('فهرس البحث المعتمد غير متوافق')
        for name,digest in meta['sha256'].items():
            if train.sha256(candidate/name)!=digest:raise ValueError('تغير ملف مرشح البحث: '+name)
        from car_model_runtime import ranked_gallery
        known=train.read_csv(train.ROOT/'الملفات_المهمة/ربط_الصور.csv')
        return ranked_gallery(gallery,g@q[0],train.sha256(Path(path)),known,5)
    r=train.load_json(train.RUN/'search/result.json')
    from car_model_runtime import ranked_gallery
    known=train.read_csv(train.ROOT/'الملفات_المهمة/ربط_الصور.csv')
    if r['accepted']:
        if f is None:f=extract([path])[0]
        kind=r['selected']['kind'];full=train.norm(f['full'][None]);crop=train.norm(f['crop'][None]);base=train.norm(f['base'][None])
        q=full if kind=='dino' else train.norm(np.column_stack([full,base] if kind=='fusion' else [full,crop,base]))
        if kind=='both_whiten':q=train.norm(joblib.load(train.RUN/'search/pca.joblib').transform(q))
        gallery=train.read_csv(train.RUN/'search/gallery.csv');g=np.load(train.RUN/'search/gallery.npy',mmap_mode='r')
    else:
        from rescue_six_predict import extract as old_extract
        q=old_extract(path)['brand'];gallery=train.read_csv(train.old.RUN/'search/gallery.csv');g=np.load(train.old.RUN/'search/gallery_embeddings.npy',mmap_mode='r')
    return ranked_gallery(gallery,g@q[0],train.sha256(Path(path)),known,5)

def main():
    if hasattr(sys.stdout,'reconfigure'):sys.stdout.reconfigure(encoding='utf-8')
    p=argparse.ArgumentParser();p.add_argument('task',choices=train.TASKS+['search']);p.add_argument('--images',nargs='+',required=True);a=p.parse_args();train.guard()
    lock=train.load_json(train.RUN/'final_lock.json')
    for path,h in lock['artifacts'].items():assert train.sha256(train.RUN/path)==h,path
    for path,h in lock['sources'].items():assert train.sha256(train.ROOT/path)==h,path
    if a.task=='search':
        if len(a.images)!=1:raise ValueError('البحث الحالي يقبل صورة استعلام واحدة')
        result={'النتائج':search(a.images[0]),'تنبيه':'تشابه بصري تجريبي؛ الإعلانات تاريخية'}
    else:
        ps=[];cs=None
        accepted=train.load_json(train.RUN/a.task/'result.json')['accepted']
        ff=extract(a.images) if accepted else [None]*len(a.images)
        for path,f in zip(a.images,ff):
            pp,cc=predict(a.task,path,f);ps.append(pp[0]);cs=cc
        avg=np.mean(ps,axis=0);ix=np.argsort(-avg)[:3]
        result={'الحالة':'تجريبي','عدد_الصور':len(ps),'طريقة_التجميع':'متوسط الاحتمالات؛ يفترض أن الصور للسيارة نفسها','النتائج':[{'الفئة':cs[i],'درجة_غير_معايرة':float(avg[i])} for i in ix],'تنبيه':'لم يختبر خارجيًا، ولا يكشف الأضرار أو السعر من الصورة'}
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
