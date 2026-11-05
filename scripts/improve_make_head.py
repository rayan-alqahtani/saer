"""تحسين رجوع الماركة بتمثيل الموديل المجمد؛ لا إعادة ضبط للشبكة."""
from improve_common import *
from improve_visual import baseline_probs,route_probs,select_rejection_threshold
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from scipy.special import softmax


def head_probs(features,head):
    return head['model'].predict_proba(head['scaler'].transform(features))


def make_features(rows):
    index=read_csv(session()/'search/finetuned_index.csv');lookup={r['sha256']:i for i,r in enumerate(index)}
    if all(r.get('sha256') in lookup and 'external_path' not in r for r in rows):
        array=np.load(session()/'search/finetuned_features.npy',mmap_mode='r');return np.asarray(array[[lookup[r['sha256']] for r in rows]])
    from improve_search import tuned_features
    return tuned_features(rows)


def refine():
    guard();s=session()
    if not (s/'search/finetuned_features.npy').exists():
        experiment('make','adapted_independent_head','لم تتوفر تمثيلات مضبوطة؛ لم أدع تدريبًا',0,0,False,{'not_run':True});return
    tr=compact_rows('make','train');va=compact_rows('make','val');x=make_features(tr);xv=make_features(va)
    y=np.asarray([r['label_make'] for r in tr]);yv=np.asarray([r['label_make'] for r in va]);old,classes=baseline_probs('make',va)
    prior=load_json(s/'make/selected.json');best=prior['val_macro_F1'];bestp=np.load(s/'make/val_probabilities.npy');chosen=prior
    pc=np.load(s/'model/val_probabilities.npy')
    # val الموديل جزء من val الماركة؛ نحتاج جميع صور الماركة للتوجيه.
    from improve_visual import predict_selected
    pc,model_classes=predict_selected('model',va)
    scaler=StandardScaler().fit(x);counts=Counter(r['group_id'] for r in tr)
    weights=np.asarray([1/counts[r['group_id']] for r in tr]);weights/=weights.mean()
    independent_best=None;independent_score=score(yv,old,classes)['macro_F1']
    for C in [.1,1.,10.]:
        for weight in [None,'balanced']:
            model=LogisticRegression(C=C,class_weight=weight,max_iter=600,solver='lbfgs').fit(scaler.transform(x),y,sample_weight=weights)
            p=model.predict_proba(scaler.transform(xv));m=score(yv,p,classes)
            keep=m['macro_F1']>independent_score
            experiment('make',f'adapted_head_C{C}_{weight}','تدريب رأس الماركة على تمثيل الموديل المجمد؛ لا تعديل rec_label',
                independent_score,m['macro_F1'],keep,{'independent_candidate_pool':True,'C':C,'class_weight':weight})
            if keep:independent_score=m['macro_F1'];independent_best={'model':model,'scaler':scaler,'classes':classes,'C':C,'class_weight':weight};independent_p=p
    head=independent_best
    if head is None:return
    for threshold in [0.,.2,.4,.6,.8,.95,1.01]:
        p=route_probs(independent_p,pc,model_classes,classes,threshold,'hard');m=score(yv,p,classes);keep=m['macro_F1']>best+1e-8
        experiment('make',f'adapted_fallback_gate_{threshold}','تغيير مصنف الرجوع المستقل إلى الرأس المحسن فقط مع إعادة اختيار عتبة val',best,m['macro_F1'],keep,m)
        if keep:
            best=m['macro_F1'];bestp=p;chosen={'kind':'route','threshold':threshold,'method':'hard','model_source':'improved',
                'independent_source':'adapted_head','classes':classes,'val_macro_F1':best}
    joblib.dump(head,s/'make/adapted_head.joblib',compress=3)
    if chosen.get('independent_source')=='adapted_head':
        conf=np.where(pc.max(1)>=chosen['threshold'],pc.max(1),independent_p.max(1));decision=np.zeros_like(bestp)
        for i,j in enumerate(bestp.argmax(1)):
            decision[i]=(1-conf[i])/(len(classes)-1);decision[i,j]=conf[i]
        chosen['rejection']=select_rejection_threshold(yv,decision,classes)
    save_json(s/'make/selected.json',chosen);np.save(s/'make/val_probabilities.npy',bestp)
    class_report('make','val',va,bestp,classes,np.asarray(classes)[old.argmax(1)]);guard()


if __name__=='__main__':refine()
