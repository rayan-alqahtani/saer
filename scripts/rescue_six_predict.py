"""تشغيل حزمة المرشحين المقبولين على صورة مجهولة أو خصائص سعر، دون تعديل المؤشرات."""
import argparse,json
import numpy as np
import joblib
from rescue_six_20261025 import ROOT,RUN,OLD,DUAL,BRAND,BASE,rich_color,guard,load_json

def extract(path):
    import torch
    from PIL import Image,ImageOps
    from torchvision.models import ResNet18_Weights
    from improve_visual import load_base
    from brand_final_round import BrandNet
    from ml_visual import color_features
    from pathlib import Path
    path=Path(path)
    if not path.is_file() or path.stat().st_size>15*1024*1024:raise ValueError('صورة مفقودة أو أكبر من 15 ميجابايت')
    torch.set_num_threads(2)
    with Image.open(path) as src:
        if src.format not in {'JPEG','PNG','WEBP'}:raise ValueError('صيغة صورة غير مدعومة')
        if src.width*src.height>40000000:raise ValueError('أبعاد صورة كبيرة')
        im=ImageOps.exif_transpose(src).convert('RGB')
    x=ResNet18_Weights.IMAGENET1K_V1.transforms()(im)[None]
    b=load_base();obj=torch.load(BRAND/'brand_model.pt',map_location='cpu',weights_only=True)
    net=BrandNet(len(obj['classes']));net.load_state_dict(obj['state_dict']);net.eval()
    with torch.inference_mode():
        stage=b.layer2(b.layer1(b.maxpool(b.relu(b.bn1(b.conv1(x))))))
        spatial=torch.nn.functional.adaptive_avg_pool2d(stage,(2,2)).flatten(1).numpy()
        base=torch.nn.functional.normalize(b.avgpool(b.layer4(b.layer3(stage))).flatten(1),dim=1).numpy()
        bb=net.backbone;brand=torch.nn.functional.normalize(bb.avgpool(bb.layer4(bb.layer3(stage))).flatten(1),dim=1).numpy()
        p=net(x).softmax(1)
        if obj['tta']:p=(p+net(x.flip(-1)).softmax(1))/2
    return {'base':base,'brand':brand,'spatial':np.column_stack([base,spatial]),
            'color':np.column_stack([base,color_features(im)[None],rich_color(path)[None]]),
            'brand_probs':p.numpy(),'brand_classes':obj['classes']}

def dual_head(task,feat):
    from retrain_dual_encoder_six import probabilities
    art=joblib.load(DUAL/task/'head.joblib')
    x=np.column_stack([feat['base'],feat['brand']])
    return probabilities(art['model'],art['scaler'],x,art['classes'],art['tau']),art['classes']

def predict_image(task,path,candidate=False,feat=None):
    from improve_visual import predict_selected
    feat=extract(path) if feat is None else feat
    rows=[{'external_path':str(path),'image_uid':'external_query','listing_uid':'external_query'}]
    accepted=load_json(RUN/task/'result.json')['accepted']
    if task=='make':ref=feat['brand_probs'];classes=feat['brand_classes']
    else:
        ref,classes=predict_selected(task,rows)
        if task=='model':
            p,cs=dual_head('model',feat);assert list(cs)==list(classes);ref=.25*ref+.75*p
    if not (accepted or candidate):return ref,classes
    if task=='body':
        pm,mc=dual_head('model',feat);pb,bc=dual_head('body',feat);assert list(classes)==list(bc)
        transition=np.load(RUN/'body/transition.npy');p=.75*ref+.25*(.25*pb+.75*(pm@transition))
    else:
        a=joblib.load(RUN/task/'head.joblib');assert list(classes)==list(a['classes'])
        p=a['model'].predict_proba(a['scaler'].transform(feat[a['feature_kind']]))
        w=1. if task=='color' else .25;p=(1-w)*ref+w*p
    return p,classes

def predict_price(row,candidate=False):
    if not isinstance(row,dict):raise ValueError('مدخل السعر يجب أن يكون كائن حقول')
    if row.get('condition','used')!='used':raise ValueError('نموذج السعر للمستعمل فقط')
    if candidate or load_json(RUN/'price/result.json')['accepted']:
        m=joblib.load(RUN/'price/candidate.joblib')
    else:
        p=load_json(ROOT/'models/price_current.json');m=joblib.load(ROOT/p['run_dir']/'model.joblib')['model']
    if not candidate:
        from car_model_runtime import validate_price
        row=validate_price(row,set(m.features.model_counts))
    return float(m.predict([row])[0])

def main():
    from pathlib import Path
    ap=argparse.ArgumentParser();ap.add_argument('task',choices=['make','model','color','body','price','search']);ap.add_argument('--input',required=True,type=Path);a=ap.parse_args();guard()
    from rescue_six_20261025 import sha256
    lock=load_json(RUN/'final_lock.json')
    for name,digest in lock['artifacts'].items():
        if sha256(RUN/name)!=digest:raise ValueError('تغير ملف الحزمة المقفلة: '+name)
    for name,digest in lock['sources'].items():
        if sha256(ROOT/name)!=digest:raise ValueError('تغير كود الحزمة المقفلة: '+name)
    if a.task=='price':
        result={'سعر_الإعلان_ريال':predict_price(json.loads(a.input.read_text(encoding='utf-8-sig'))),'تنبيه':'سعر إعلان للمستعمل، وليس سعر بيع مضمونًا'}
    elif a.task=='search':
        if not load_json(RUN/'search/result.json')['accepted']:
            from improve_predict import search_result
            result=search_result(a.input,5)
        else:
            from rescue_six_20261025 import read_csv,sha256
            rows=read_csv(RUN/'search/gallery.csv');e=np.load(RUN/'search/gallery_embeddings.npy',mmap_mode='r');q=extract(a.input)['brand'][0]
            from car_model_runtime import ranked_gallery
            known=read_csv(ROOT/'الملفات_المهمة/ربط_الصور.csv')
            found=ranked_gallery(rows,e@q,sha256(a.input),known,5)
            result={'النتائج':found,'تنبيه':'التشابه ليس احتمال صحة معايرًا'}
    else:
        p,classes=predict_image(a.task,a.input);order=np.argsort(-p[0])[:3]
        result={'الحالة':'تجريبي','اقتراحات':[{'الفئة':classes[i],'درجة_غير_معايرة':float(p[0,i])} for i in order],'تنبيه':'لم يختبر خارجيًا؛ ليست درجات ثقة مضمونة'}
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
