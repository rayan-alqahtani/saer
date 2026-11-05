"""تشغيل حزمة النماذج دون جداول التدريب؛ جميع الملفات محلية."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ.setdefault(key,'4')
import argparse
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
import joblib
from PIL import Image,ImageOps,UnidentifiedImageError
from scipy import sparse
from scipy.special import softmax

HERE=Path(__file__).resolve().parent


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def check_files():
    manifest=read_json(HERE/'integrity_manifest.json')
    for name,digest in manifest['files_sha256'].items():
        path=HERE/name
        if not path.is_file() or hashlib.file_digest(path.open('rb'),'sha256').hexdigest()!=digest:
            raise ValueError('ملف الحزمة مفقود أو تغير: '+name)
    return manifest


def load(task):
    data=joblib.load(HERE/(task+'.joblib'))
    identity=read_json(HERE/'dataset_identity.json')
    for key in ('dataset_version','dataset_fingerprint','split_fingerprint'):
        if data['metadata'].get(key)!=identity[key]: raise ValueError('هوية النموذج لا تطابق الحزمة')
    return data


def price(row):
    data=load('price')
    if row.get('condition','used')!='used': raise ValueError('نموذج السعر مخصص للمستعمل فقط')
    f=data['vocabulary']; mk=row['make']; name=mk+'|'+row['model_std']; trim=row['trim_std'].strip().lower()
    year=float(row['year']); km=float(row['km_mileage'])
    if not np.isfinite(year) or not np.isfinite(km) or km<0 or not 1950<=year<=2027: raise ValueError('سنة أو ممشى غير صالح')
    cats=[[mk,name if f['model_counts'].get(name,0)>=5 else mk+'|other',
           trim if trim in f['trim_keep'] else '__other_or_missing__',
           row['city'] if row['city'] in f['city_keep'] else '__other__',
           row['transmission'] or '__missing__',row['type_fuel'] or '__missing__']]
    nums=np.asarray([[2026-year,np.log1p(km)]])
    if data['kind']=='dummy': pred=data['estimator'].predict(nums)[0]
    elif data['kind']=='hierarchical_median':
        pred=data['global_median']
        for keys,values in data['medians']:
            key=tuple(row[k] for k in keys)
            if key in values: pred=values[key]; break
    else:
        enc=data['encoder'].transform(np.asarray(cats,dtype=object))
        x=np.column_stack([enc,nums]) if data['kind']=='hgb' else sparse.hstack([enc,nums],format='csr')
        pred=np.exp(data['estimator'].predict(x)[0])
    return {'سعر_الإعلان_المقدر_ريال':round(float(pred),2),'الحالة':data['metadata']['status'],
            'تنبيه':'سعر إعلان تاريخي للمستعمل، وليس سعر بيع مضمونًا؛ لا نطاق ثقة معاير'}


def color_features(im):
    w,h=im.size; out=[]
    for crop in [im,im.crop((int(.18*w),int(.2*h),int(.82*w),int(.85*h)))]:
        a=np.asarray(crop.resize((96,72)).convert('HSV'),dtype=float)/255.
        for channel,bins in [(0,24),(1,16),(2,16)]:
            out.extend(np.histogram(a[:,:,channel],bins=bins,range=(0,1))[0]/a.shape[0]/a.shape[1])
        rgb=np.asarray(crop.resize((96,72)),dtype=float)/255.
        out.extend(np.mean(rgb,axis=(0,1)));out.extend(np.std(rgb,axis=(0,1)))
    return np.asarray(out,dtype=np.float32)


def embed(photo):
    import torch
    from torchvision.models import resnet18,ResNet18_Weights
    torch.set_num_threads(4)
    model=resnet18(weights=None)
    model.load_state_dict(torch.load(HERE/'encoder_weights.pth',map_location='cpu',weights_only=True))
    model.fc=torch.nn.Identity();model.eval();transform=ResNet18_Weights.IMAGENET1K_V1.transforms()
    try:
        with Image.open(photo) as src:
            if src.width*src.height>40_000_000: raise ValueError('أبعاد الصورة تتجاوز الحد')
            im=ImageOps.exif_transpose(src).convert('RGB');c=color_features(im);tensor=transform(im)
    except (OSError,UnidentifiedImageError) as exc: raise ValueError('الصورة غير صالحة') from exc
    with torch.inference_mode(): e=model(tensor[None]).numpy()[0]
    e/=max(float(np.linalg.norm(e)),1e-12)
    return e,c


def classify(task,e,c):
    a=load(task)
    x=e[None] if a['mode']=='embeddings' else c[None] if a['mode']=='color' else np.column_stack([e[None],c[None]])
    scores=a['model'].decision_function(a['scaler'].transform(x))
    if scores.ndim==1: scores=np.column_stack([-scores/2,scores/2])
    p=softmax(scores/a['temperature'],axis=1)[0]; order=np.argsort(-p)[:3]
    accepted=bool(p[order[0]]>=a['rejection']['threshold'])
    return {'الحالة':'تجريبي','النتيجة':a['classes'][order[0]] if accepted else 'غير متأكد','مقبول':accepted,
            'الاقتراحات':[{'الفئة':a['classes'][i],'درجة_النموذج':round(float(p[i]),5)} for i in order],
            'تنبيه':'مقاييس مصدر واحد؛ لا ضمان لصحة صور خارج النطاق'}


def search(photo,e,k):
    if not 1<=k<=20: raise ValueError('عدد النتائج من 1 إلى 20')
    rows=read_json(HERE/'search_index.json'); vectors=np.load(HERE/'search_embeddings.npy',mmap_mode='r')
    digest=hashlib.file_digest(Path(photo).open('rb'),'sha256').hexdigest()
    excluded={r['group_id'] for r in rows if r['sha256']==digest}
    scores=vectors@e; order=np.argsort(-scores); seen=set(); results=[]
    for i in order:
        r=rows[i]
        if r['group_id'] in excluded or r['group_id'] in seen or r['sha256']==digest: continue
        seen.add(r['group_id']);results.append({'معرف_الإعلان':r['listing_uid'],'الصورة':r['relative_path'],'التشابه':round(float(scores[i]),5)})
        if len(results)==k: break
    return {'الحالة':'تجريبي','النتائج':results,'تنبيه':'تشابه وليس احتمال صحة؛ الصور المشار إليها في مجموعة البيانات المحلية الأصلية، وليست مرفقة بهذه الحزمة'}


def run(task,input_path=None,k=5):
    check_files()
    if task=='status': return read_json(HERE/'model_status.json')
    if task=='price': return price(read_json(input_path))
    e,c=embed(input_path)
    if task=='search': return search(input_path,e,k)
    if task=='recognize':
        make=classify('make',e,c);model=classify('model',e,c)
        matched=make['مقبول'] and model['مقبول'] and model['النتيجة'].split('|')[0]==make['النتيجة']
        return {'الماركة':make,'الموديل':model,'المسار':'model_and_make' if matched else 'make_only' if make['مقبول'] else 'unavailable',
                'الموديل_المعتمد':model['النتيجة'] if matched else 'غير متأكد','نطاق_الموديل':'59 فئة فقط'}
    return classify(task,e,c)


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    ap=argparse.ArgumentParser();ap.add_argument('task',choices=['price','make','model','recognize','color','body','search','status'])
    ap.add_argument('--input',type=Path);ap.add_argument('--k',type=int,default=5);ap.add_argument('--output',type=Path)
    args=ap.parse_args()
    try:
        if args.task!='status' and (not args.input or not args.input.is_file()): raise ValueError('حدد ملف إدخال موجودًا')
        out={'نجاح':True,'النتيجة':run(args.task,args.input,args.k)};code=0
    except (ValueError,KeyError,FileNotFoundError,json.JSONDecodeError) as exc:
        out={'نجاح':False,'الخطأ':str(exc)};code=1
    text=json.dumps(out,ensure_ascii=False,indent=2)
    if args.output: args.output.write_text(text,encoding='utf-8')
    print(text);return code


if __name__=='__main__': raise SystemExit(main())
