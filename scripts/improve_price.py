"""اختبارات السعر: GroupKFold داخل التدريب، اختيار على val فقط."""
from improve_common import *
from ml_price import PriceFeatures,FEATURES,metrics
import argparse,time
import joblib
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import OrdinalEncoder,OneHotEncoder
from sklearn.ensemble import HistGradientBoostingRegressor,ExtraTreesRegressor
from scipy import sparse


class PriceCandidate:
    def __init__(self,config):self.config=config
    def fit(self,rows):
        self.features=PriceFeatures().fit(rows);cats,nums=self.features.transform(rows)
        cfg=self.config.copy();kind=cfg.pop('kind');weighted=cfg.pop('group_weight',False);self.kind=kind
        y=np.log(np.asarray([float(r['sar_price']) for r in rows]))
        if kind=='hgb':
            self.encoder=OrdinalEncoder(handle_unknown='use_encoded_value',unknown_value=np.nan)
            x=np.column_stack([self.encoder.fit_transform(cats),nums])
            self.model=HistGradientBoostingRegressor(random_state=42,early_stopping=False,
                categorical_features=[True]*6+[False]*2,monotonic_cst=[0]*6+[-1,-1],**cfg)
        else:
            self.encoder=OneHotEncoder(handle_unknown='ignore');x=sparse.hstack([self.encoder.fit_transform(cats),nums],format='csr')
            self.model=ExtraTreesRegressor(n_jobs=4,random_state=42,**cfg)
        counts=Counter(r['group_id'] for r in rows)
        weight=np.asarray([1/counts[r['group_id']] for r in rows]) if weighted else None
        if weight is not None:weight/=weight.mean()
        self.model.fit(x,y,sample_weight=weight);return self
    def predict(self,rows):
        cats,nums=self.features.transform(rows);encoded=self.encoder.transform(cats)
        x=np.column_stack([encoded,nums]) if self.kind=='hgb' else sparse.hstack([encoded,nums],format='csr')
        return np.exp(self.model.predict(x))


class MedianModel:
    def fit(self,rows):
        groups=defaultdict(list);makes=defaultdict(list)
        for r in rows:groups[(r['make'],r['model_std'])].append(float(r['sar_price']));makes[r['make']].append(float(r['sar_price']))
        self.by_model={k:float(np.median(v)) for k,v in groups.items()};self.by_make={k:float(np.median(v)) for k,v in makes.items()}
        self.global_median=float(np.median([float(r['sar_price']) for r in rows]));return self
    def predict(self,rows):return np.asarray([self.by_model.get((r['make'],r['model_std']),self.by_make.get(r['make'],self.global_median)) for r in rows])


def price_metrics(rows,p):
    y=np.asarray([float(r['sar_price']) for r in rows]);m=metrics(y,p);m['MAPE_percent']=float(np.mean(abs(y-p)/y)*100)
    groups=sorted({r['group_id'] for r in rows});stats=np.zeros((len(groups),3));lookup={g:i for i,g in enumerate(groups)}
    for r,a,b in zip(rows,y,p):stats[lookup[r['group_id']]]+=[1,abs(a-b),abs(a-b)/a*100]
    weights=np.random.default_rng(42).multinomial(len(groups),np.full(len(groups),1/len(groups)),size=2000)
    boot=weights@stats
    m['ci95']={'MAE_SAR':np.quantile(boot[:,1]/boot[:,0],[.025,.975]).tolist(),
                'MAPE_percent':np.quantile(boot[:,2]/boot[:,0],[.025,.975]).tolist(),
                'bootstrap_unit':'group_id','groups':len(groups),'repetitions':2000}
    return m,boot[:,1]/boot[:,0]


def fit():
    guard();folder=session()/'price';tr=compact_rows('price','train');va=compact_rows('price','val')
    if not all(r['condition']=='used' for r in tr+va):raise ValueError('دخل الجديد نطاق السعر')
    old=joblib.load(original('price')/'model.joblib')['model'];oldp=old.predict(va);before=price_metrics(va,oldp)[0]
    best=before['MAE_SAR'];selected={'kind':'original','task':'price','val_MAE_SAR':best};best_model=old
    save_json(folder/'val_before_metrics.json',before)
    configs=[]
    for loss in ['absolute_error','squared_error']:
        for leaves in [15,31]:
            configs.append({'kind':'hgb','loss':loss,'learning_rate':.035,'max_iter':700,'max_leaf_nodes':leaves,'min_samples_leaf':10,'l2_regularization':2.})
    configs.append({'kind':'hgb','loss':'absolute_error','learning_rate':.035,'max_iter':700,'max_leaf_nodes':15,'min_samples_leaf':10,'l2_regularization':2.,'group_weight':True})
    configs.append({'kind':'extra','n_estimators':500,'min_samples_leaf':1,'max_features':.8})
    splitter=GroupKFold(n_splits=5);g=np.asarray([r['group_id'] for r in tr]);cv_rows=[]
    for n,cfg in enumerate(configs):
        guard();start=time.perf_counter();oof=np.empty(len(tr))
        for fold,(train_idx,valid_idx) in enumerate(splitter.split(np.arange(len(tr)),groups=g)):
            if set(g[train_idx])&set(g[valid_idx]):raise ValueError('تسرب في GroupKFold')
            model=PriceCandidate(cfg).fit([tr[i] for i in train_idx]);oof[valid_idx]=model.predict([tr[i] for i in valid_idx])
            print('PRICE',n,'fold',fold+1,'done',flush=True)
        cv=price_metrics(tr,oof)[0];model=PriceCandidate(cfg).fit(tr);p=model.predict(va);val=price_metrics(va,p)[0]
        keep=val['MAE_SAR']<best;cv_rows.append({'candidate':n,'config':cfg,'group_cv':cv,'val':val})
        save_json(folder/'group_cv_results.json',cv_rows)
        experiment('price',f'candidate{n}','مقارنة مرشح log-price مع GroupKFold تدريبي؛ الاختيار النهائي val فقط',best,val['MAE_SAR'],keep,
            {'config':cfg,'cv_MAE':cv['MAE_SAR'],'val_MAPE':val['MAPE_percent'],'seconds':time.perf_counter()-start})
        if keep:
            best=val['MAE_SAR'];best_model=model;selected={'kind':'candidate','task':'price','config':cfg,'val_MAE_SAR':best}
    joblib.dump(best_model,folder/'best_model.joblib',compress=3)
    joblib.dump(MedianModel().fit(tr),folder/'median_baseline.joblib',compress=3)
    save_json(folder/'selected.json',selected)
    save_json(folder/'val_metrics.json',price_metrics(va,best_model.predict(va))[0])
    save_json(folder/'val_median_baseline.json',price_metrics(va,MedianModel().fit(tr).predict(va))[0]);guard()


if __name__=='__main__':
    # اسم ثابت لفك النماذج في عمليات مستقلة.
    sys.modules['improve_price']=sys.modules[__name__];PriceCandidate.__module__='improve_price';MedianModel.__module__='improve_price'
    fit()
