"""استدلال محلي للاختيارات الجديدة، دون تبديل النماذج الحالية."""
from improve_common import *
from improve_visual import predict_selected,make_prediction,baseline_probs
from ml_visual import load_encoder,image_embedding,embedding_data
import argparse,joblib


def image_result(task,path):
    rows=[{'external_path':str(path.resolve()),'image_uid':'query','listing_uid':'query'}]
    if task=='make':p,classes,confidence=make_prediction(rows,return_confidence=True)
    else:p,classes=predict_selected(task,rows);confidence=p.max(1)
    order=np.argsort(-p[0])[:3];cfg=load_json(session()/task/'selected.json')
    threshold=cfg.get('rejection',{}).get('threshold',1.01)
    accepted=bool(confidence[0]>=threshold)
    return {'الحالة':'تجريبي','النتيجة':classes[order[0]] if accepted else 'غير متأكد',
        'اقتراحات':[{'الفئة':classes[i],'درجة_قرار_غير_مضمونة':round(float(p[0,i]),5)} for i in order],
        'تنبيه':('درجات الماركة hard قرارات فئوية وليست احتمال صحة؛ ' if task=='make' else '')+'لا تعميم مثبت خارج المصدر'}


def search_result(path,k):
    if not 1<=k<=20:raise ValueError('عدد النتائج بين 1 و20')
    cfg=load_json(session()/'search/selected.json');rows=compact_rows('search','train')
    if cfg['kind']=='finetuned_cosine':
        from improve_search import tuned_features
        q=tuned_features([{'external_path':str(path.resolve())}])[0]
        indexed=read_csv(session()/'search/finetuned_index.csv');lookup={r['sha256']:i for i,r in enumerate(indexed)}
        e=np.load(session()/'search/finetuned_features.npy',mmap_mode='r')
    else:
        encoder,tf=load_encoder(original('embeddings')/'encoder_weights.pth');q,c=image_embedding(path,encoder,tf)
        cache,cm,lookup,e,col=embedding_data()
    allrows=compact_rows('search');digest=sha256(path);excluded={r['group_id'] for r in allrows if r['sha256']==digest}
    vectors=e[[lookup[r['sha256']] for r in rows]];sim=vectors@q;order=np.argsort(-sim);seen=set();result=[]
    for i in order:
        r=rows[i]
        if r['group_id'] in excluded or r['group_id'] in seen or r['sha256']==digest:continue
        seen.add(r['group_id']);result.append({'معرف_الإعلان':r['listing_uid'],'الصورة':r['relative_path'],'التشابه':float(sim[i])})
        if len(result)==k:break
    return {'النتائج':result,'تنبيه':'المعرض تدريب فقط؛ استبعدت المجموعة والإعلان والصورة المطابقة. التشابه ليس احتمال صحة'}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('task',choices=['status','price','make','model','color','body','search'])
    parser.add_argument('--input',type=Path);parser.add_argument('--output',type=Path);parser.add_argument('--k',type=int,default=5);args=parser.parse_args()
    try:
        guard()
        if args.task=='status':
            result={t:load_json(session()/t/'selected.json',{'status':'غير متوفر'}) for t in ['make','model','color','body','price','search']}
        else:
            if not args.input or not args.input.is_file():raise ValueError('حدد ملف إدخال موجودًا')
            if args.task=='price':
                row=load_json(args.input)
                if row.get('condition','used')!='used':raise ValueError('السعر للمستعمل فقط')
                pred=float(joblib.load(session()/'price/best_model.joblib').predict([row])[0])
                result={'سعر_الإعلان_المقدر_ريال':round(pred,2),'تنبيه':'تقدير إعلان، لا سعر بيع مضمون ولا فاصل تنبؤ فردي'}
            elif args.task=='search':result=search_result(args.input,args.k)
            else:result=image_result(args.task,args.input)
        out={'نجاح':True,'النتيجة':result};code=0
    except (ValueError,KeyError,FileNotFoundError,OSError) as exc:out={'نجاح':False,'الخطأ':str(exc)};code=1
    if args.output:save_json(args.output,out)
    print(json.dumps(out,ensure_ascii=False,indent=2));return code


if __name__=='__main__':raise SystemExit(main())
