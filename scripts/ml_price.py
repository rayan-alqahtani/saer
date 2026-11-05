"""تدريب السعر على التقسيم الثابت؛ اختيار بالتحقق واختبار مرة بعد الاختيار."""
from ml_common import *
import argparse
import time
from collections import Counter, defaultdict
import numpy as np
import joblib
from sklearn.dummy import DummyRegressor
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.ensemble import HistGradientBoostingRegressor, ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy import sparse

FEATURES = ['make','model_std','year','km_mileage','trim_std','city','transmission','type_fuel']
CATEGORICAL = ['make','model_std','trim_std','city','transmission','type_fuel']


class PriceFeatures:
    def fit(self, rows):
        self.model_counts = Counter(r['make']+'|'+r['model_std'] for r in rows)
        self.trim_keep = {t for t,n in Counter(r['trim_std'].strip().lower() for r in rows).items() if t and n >= 5}
        self.city_keep = {c for c,n in Counter(r['city'] for r in rows).items() if n >= 3}
        return self

    def transform(self, rows):
        cats, nums = [], []
        for r in rows:
            # لا يسمح أي عمود من خارج القائمة البيضاء بالتأثير على السمات.
            for k in FEATURES:
                if k not in r: raise ValueError('حقل مطلوب: '+k)
            y, km = float(r['year']), float(r['km_mileage'])
            if not np.isfinite(y) or not np.isfinite(km) or km < 0 or y < 1950 or y > 2027:
                raise ValueError('سنة أو ممشى غير صالح')
            mk, model, trim = r['make'], r['make']+'|'+r['model_std'], r['trim_std'].strip().lower()
            cats.append([mk, model if self.model_counts[model] >= 5 else mk+'|other',
                         trim if trim in self.trim_keep else '__other_or_missing__',
                         r['city'] if r['city'] in self.city_keep else '__other__',
                         r['transmission'] or '__missing__', r['type_fuel'] or '__missing__'])
            nums.append([2026-y, np.log1p(km)])
        return np.asarray(cats, dtype=object), np.asarray(nums, dtype=float)


class PriceModel:
    def __init__(self, kind, config=None): self.kind, self.config = kind, config or {}

    def fit(self, rows):
        self.features = PriceFeatures().fit(rows)
        cats, nums = self.features.transform(rows)
        y = np.asarray([float(r['sar_price']) for r in rows])
        if np.any(y<=0): raise ValueError('سعر غير موجب')
        if self.kind == 'dummy':
            self.estimator = DummyRegressor(strategy='median').fit(nums, y)
        elif self.kind == 'hierarchical_median':
            self.medians = []
            for keys, minimum in [(('make','model_std','year'),3),(('make','model_std'),3),(('make',),1)]:
                groups = defaultdict(list)
                for r in rows: groups[tuple(r[k] for k in keys)].append(float(r['sar_price']))
                self.medians.append((keys,{k:float(np.median(v)) for k,v in groups.items() if len(v)>=minimum}))
            self.global_median=float(np.median(y))
        else:
            if self.kind == 'hgb':
                self.encoder=OrdinalEncoder(handle_unknown='use_encoded_value',unknown_value=np.nan,encoded_missing_value=np.nan)
                x=np.column_stack([self.encoder.fit_transform(cats),nums])
                self.estimator=HistGradientBoostingRegressor(loss='absolute_error',early_stopping=False,
                    random_state=SEED,categorical_features=[True]*6+[False]*2,
                    monotonic_cst=[0]*6+[-1,-1],**self.config)
            elif self.kind == 'extra_trees':
                self.encoder=OneHotEncoder(handle_unknown='ignore',sparse_output=True)
                x=sparse.hstack([self.encoder.fit_transform(cats),nums],format='csr')
                self.estimator=ExtraTreesRegressor(n_estimators=350,n_jobs=4,random_state=SEED,**self.config)
            else: raise ValueError(self.kind)
            self.estimator.fit(x,np.log(y))
        return self

    def predict(self, rows):
        cats, nums = self.features.transform(rows)
        if self.kind == 'dummy': return self.estimator.predict(nums)
        if self.kind == 'hierarchical_median':
            out=[]
            for r in rows:
                value=self.global_median
                for keys,values in self.medians:
                    key=tuple(r[k] for k in keys)
                    if key in values: value=values[key]; break
                out.append(value)
            return np.asarray(out)
        encoded=self.encoder.transform(cats)
        x=np.column_stack([encoded,nums]) if self.kind=='hgb' else sparse.hstack([encoded,nums],format='csr')
        return np.exp(self.estimator.predict(x))


def metrics(y,p):
    y,p=np.asarray(y),np.asarray(p)
    return {'n':len(y),'MAE_SAR':float(mean_absolute_error(y,p)),
            'RMSE_SAR':float(np.sqrt(mean_squared_error(y,p))),
            'R2':float(r2_score(y,p)) if len(y)>1 else None,
            'median_APE_percent':float(np.median(np.abs(y-p)/y)*100),
            'within_20_percent':float(np.mean(np.abs(y-p)/y<=.2)*100)}


def train():
    folder,meta=new_run('price')
    rows=read_csv(PROCESSED/'price_dataset.csv')
    assert_split_integrity(rows)
    assert set(FEATURES)==set(load_json(PROCESSED/'schema.json')['price_features_allowlist'])
    assert all(r['usable_price']=='True' and r['condition']=='used' for r in rows)
    tr,va,te=([r for r in rows if r['split']==s] for s in ('train','val','test'))
    log(folder,f'السعر: train={len(tr)} val={len(va)} test={len(te)}؛ الاختبار مغلق حتى الاختيار')
    candidates=[('dummy','dummy',{}),('hierarchical_median','hierarchical_median',{})]
    for leaves,leaf in [(15,10),(31,8)]:
        for iterations in [250,500]:
            candidates.append((f'hgb_l{leaves}_i{iterations}','hgb',{'max_leaf_nodes':leaves,'min_samples_leaf':leaf,
                'max_iter':iterations,'learning_rate':.05,'l2_regularization':1.}))
    for leaf in [1,3]: candidates.append((f'extra_trees_leaf{leaf}','extra_trees',{'min_samples_leaf':leaf,'max_features':1.0}))
    save_json(folder/'plan.json',{'selection_metric':'validation_MAE_SAR','candidates':candidates,
        'fit_split':'train','selection_split':'val','final_split':'test','refit_on_val':False,
        'features':FEATURES,'not_features':load_json(PROCESSED/'schema.json')['forbidden_model_features'],
        'rationale_ar':'استفدت من قاموس التدريب ووسيط الفئات والهدف اللوغاريتمي في السكربت القديم؛ أوقفت الخلط وإعادة التقسيم والتوقف المبكر العشوائي. لا يستعمل CV قديم.'})
    start=time.perf_counter(); fitted={}; results=[]
    for name,kind,config in candidates:
        t=time.perf_counter(); model=PriceModel(kind,config).fit(tr)
        m=metrics([float(r['sar_price']) for r in va],model.predict(va))
        result={'candidate':name,'kind':kind,'config':config,'validation':m,'fit_seconds':time.perf_counter()-t}
        results.append(result); fitted[name]=model
        save_json(folder/'selection.json',results)
        log(folder,f'{name}: validation MAE={m["MAE_SAR"]:.1f} SAR، {result["fit_seconds"]:.1f} ثانية')
    selected=min(results,key=lambda r:r['validation']['MAE_SAR'])
    name=selected['candidate']; model=fitted[name]
    # يجمد الاختيار قبل أول قراءة لمقاييس الاختبار، ولا يعاد التدريب بعد ذلك.
    save_json(folder/'selection_lock.json',{'selected':selected,'locked_at':now(),'test_not_evaluated_yet':True})
    artifact={'model':model,'metadata':{**meta,'features':FEATURES,'selected_candidate':name}}
    joblib.dump(artifact,folder/'model.joblib',compress=3)
    test={}; predictions=[]
    for candidate in dict.fromkeys([name,'dummy','hierarchical_median']):
        pred=fitted[candidate].predict(te)
        test[candidate]=metrics([float(r['sar_price']) for r in te],pred)
        for r,p in zip(te,pred): predictions.append({'listing_uid':r['listing_uid'],'group_id':r['group_id'],
            'split':'test','candidate':candidate,'actual_sar':r['sar_price'],'predicted_sar':float(p),'absolute_error_sar':abs(float(r['sar_price'])-p)})
    pred=model.predict(te); segments={}
    counts=Counter(r['make']+'|'+r['model_std'] for r in tr)
    masks={}
    for mk in sorted({r['make'] for r in te}): masks['make='+mk]=[i for i,r in enumerate(te) if r['make']==mk]
    for label,lo,hi in [('train_model_support_<5',0,4),('train_model_support_5_19',5,19),('train_model_support_20plus',20,99999)]:
        masks[label]=[i for i,r in enumerate(te) if lo<=counts[r['make']+'|'+r['model_std']]<=hi]
    for label,lo,hi in [('price_<50000',0,49999),('price_50000_150000',50000,150000),('price_>150000',150001,1e9)]:
        masks[label]=[i for i,r in enumerate(te) if lo<=float(r['sar_price'])<=hi]
    for label,indices in masks.items():
        if indices: segments[label]=metrics([float(te[i]['sar_price']) for i in indices],pred[indices])
    result={'selected_candidate':name,'validation_selection':selected,'test':test,'test_segments':segments,
            'split_counts':dict(Counter(r['split'] for r in rows)), 'elapsed_seconds':time.perf_counter()-start,
            'status':'ready' if test[name]['MAE_SAR']<test['dummy']['MAE_SAR'] else 'experimental',
            'target_ar':'سعر إعلان تاريخي للمستعمل، وليس سعر بيع نهائي',
            'limits_ar':['مصدر واحد وانحياز للرياض وفئات قليلة الدعم','لا تقييم خارجي أو زمني مستقل','لا نطاق ثقة معاير في هذه النسخة','مقاييس الاختبار تخص النموذج المختار فقط دون اختيار جديد بعدها']}
    save_json(folder/'metrics.json',result); save_csv(folder/'test_predictions.csv',predictions)
    meta.update(training_status='completed',selected_candidate=name,features=FEATURES,completed_at=now(),status=result['status'])
    save_json(folder/'metadata.json',meta)
    verified_inputs()
    save_json(ROOT/'models/price_current.json',{'run_dir':str(folder.relative_to(ROOT)),**{k:meta[k] for k in ('dataset_version','dataset_fingerprint','split_fingerprint')}})
    log(folder,'اكتمل السعر: '+str(result['test'][name]))
    print('PRICE_RUN='+str(folder),flush=True)


if __name__=='__main__':
    # حفظ اسم وحدة ثابت، حتى لا يعتمد فك النموذج على __main__ لجلسة التدريب.
    sys.modules['ml_price']=sys.modules[__name__]
    PriceFeatures.__module__='ml_price'
    PriceModel.__module__='ml_price'
    train()
