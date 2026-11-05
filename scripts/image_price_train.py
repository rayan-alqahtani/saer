"""تجربة مستقلة: صورة واحدة لكل إعلان، اختيار val وقفل test مرة واحدة."""
import argparse
import copy
import csv
import json
import os
import shutil
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '4')
import joblib
import numpy as np
import torch
from PIL import Image, ImageOps
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.model_selection import GroupKFold
from sklearn.base import clone
from torchvision.models import resnet18, ResNet18_Weights

from day20_common import ROOT, PROCESSED, read_csv, load_json, sha256, now
from improve_common import compare_inputs
from ml_price import metrics
from image_price_runtime import domain_score, load_image

SEED = 20261022
EMBED = ROOT / 'models/20261021T090153837145_embeddings'
ACT = ROOT / 'models/improvement_20261021T093545513846/activation_cache'
torch.set_num_threads(4)


def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def write_csv(path, rows):
    if not rows:
        Path(path).write_text('', encoding='utf-8-sig'); return
    with Path(path).open('w', encoding='utf-8-sig', newline='') as f:
        fields=list(dict.fromkeys(k for r in rows for k in r))
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)


def check_inputs(run=None):
    check, _ = compare_inputs()
    if check['critical'] or not all(v['authorized_match'] for v in check['files'].values()):
        raise ValueError('تغيرت هوية البيانات المعتمدة أو التقسيم')
    canonical=load_json(PROCESSED/'training_manifest.json')
    check['canonical_manifest_matches']=all(sha256(PROCESSED/n)==h for n,h in canonical['input_files_sha256'].items())
    if not check['canonical_manifest_matches']:raise ValueError('جداول processed لا تطابق manifest')
    if run is not None:
        p = load_json(run / 'protocol.json')
        for rel, expected in p['frozen_files'].items():
            if sha256(ROOT / rel) != expected:
                raise ValueError('تغير الملف المقفل: ' + rel)
        if sha256(run / 'selected_images.csv') != p['selection_sha256']:
            raise ValueError('تغير اختيار صور المهمة')
    return check


def choose_images(listings, images):
    """قاعدة لا تقرأ السعر أو الماركة أو السنة؛ تحتفظ بترتيب الصورة الأول المؤهل."""
    by_id = {r['listing_uid']: r for r in listings if r['usable_price'] == 'True' and r['condition'] == 'used'}
    candidates = defaultdict(list)
    for r in images:
        if r['listing_uid'] not in by_id: continue
        if r.get('is_exterior') != 'True' or r.get('content_kind') != 'vehicle_overview': continue
        if r.get('usable_visual_search') != 'True': continue
        listing = by_id[r['listing_uid']]
        if (r['split'], r['group_id']) != (listing['split'], listing['group_id']):
            raise ValueError('اختلاف قسم/مجموعة الصورة والإعلان')
        candidates[r['listing_uid']].append(r)
    return [min(candidates[uid], key=lambda r: (r['image_uid'], r['sha256'])) for uid in sorted(candidates)]


def audit_near(rows):
    """فرز ثابت بالصور فقط قبل اختيار أي نموذج؛ ليس ضمان كشف كل التكرارات."""
    ph = [int(r['phash'], 16) for r in rows]
    dh = [int(r['dhash'], 16) for r in rows]
    thumbnails = {}; pairs = []; excluded = set()
    def thumb(i):
        if i not in thumbnails:
            with Image.open(ROOT / 'الملفات_المهمة' / rows[i]['relative_path']) as im:
                thumbnails[i] = np.asarray(ImageOps.exif_transpose(im).convert('RGB').resize((64,64)), dtype=np.float32)
        return thumbnails[i]
    for i, r in enumerate(rows):
        for j in range(i + 1, len(rows)):
            s = rows[j]
            if r['group_id'] == s['group_id']: continue
            if (ph[i] ^ ph[j]).bit_count() > 8 or (dh[i] ^ dh[j]).bit_count() > 10: continue
            mae = float(np.abs(thumb(i) - thumb(j)).mean())
            if mae <= 3:
                pairs.append({'left': r['image_uid'], 'right': s['image_uid'], 'left_split': r['split'], 'right_split': s['split'], 'rgb_mae': mae})
                if r['split'] != s['split']: excluded.update((r['group_id'], s['group_id']))
    return pairs, excluded


def prepare():
    check = check_inputs()
    run = ROOT / 'models' / ('image_price_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f'))
    run.mkdir()
    listings = read_csv(PROCESSED / 'listings.csv'); images = read_csv(PROCESSED / 'images.csv')
    compact = {r['image_uid']:r for r in read_csv(ROOT / 'الملفات_المهمة/ربط_الصور.csv')}
    rows = choose_images(listings, images)
    for r in rows:
        r['relative_path'] = compact[r['image_uid']]['relative_path']
        path = ROOT / 'الملفات_المهمة' / r['relative_path']
        if sha256(path) != r['sha256']: raise ValueError('تغيرت صورة: ' + r['image_uid'])
    pairs, excluded = audit_near(rows)
    # عزل المجموعات المعروفة من المراجعة السابقة، لا تعديل split/group_id الأصليين.
    known_ids = {'external_group_b:0035US', 'external_group_b:0036US'}
    known_groups = {r['group_id'] for r in listings if r['listing_uid'] in known_ids}
    excluded |= known_groups
    rejected = [{'listing_uid':r['listing_uid'], 'group_id':r['group_id'], 'reason':'known_or_strict_near_duplicate_cross_split'} for r in rows if r['group_id'] in excluded]
    rows = [r for r in rows if r['group_id'] not in excluded]
    fields = ['image_uid','listing_uid','relative_path','sha256','split','group_id']
    selected = [{k:r[k] for k in fields} for r in rows]
    validate_rows(selected)
    write_csv(run / 'selected_images.csv', selected)
    write_csv(run / 'excluded_near_duplicates.csv', rejected)
    write_json(run / 'near_audit.json', {'pairs':pairs,'excluded_groups':sorted(excluded),
        'known_scene_listing_ids':sorted(known_ids),'rule':'pHash<=8,dHash<=10,RGB64 MAE<=3; exclude both groups across splits',
        'scope':'selected images; prior confirmed scene groups also excluded; not exhaustive proof'})
    eligible = [r for r in listings if r['usable_price'] == 'True']
    selected_ids = {r['listing_uid'] for r in selected}
    write_csv(run / 'excluded_listings.csv', [{'listing_uid':r['listing_uid'],'split':r['split'],
        'reason':'near_duplicate_exclusion' if r['group_id'] in excluded else 'no_eligible_exterior_overview'} for r in eligible if r['listing_uid'] not in selected_ids])
    frozen = {}
    for p in [PROCESSED/'listings.csv',PROCESSED/'images.csv',PROCESSED/'training_manifest.json',
              EMBED/'encoder_weights.pth', EMBED/'embeddings.npy', EMBED/'image_index.csv',
              *sorted((ROOT/'models').glob('*_current.json'))]:
        frozen[p.relative_to(ROOT).as_posix()] = sha256(p)
    protocol = {'created_at':now(),'seed':SEED,'input_check':check,'frozen_files':frozen,
        'selection_sha256':sha256(run/'selected_images.csv'), 'splits':dict(Counter(r['split'] for r in rows)),
        'groups':{s:len({r['group_id'] for r in rows if r['split']==s}) for s in ['train','val','test']},
        'one_image_per_listing':True,'image_selection':'first image_uid among exterior/vehicle_overview/usable_visual_search; independent of target',
        'price_features':'pixels only; no metadata, names, IDs, flags, year or mileage as predictive features',
        'scope':'used advertised price only', 'selection_metric':'val MAE in SAR, all eligible val samples before rejection',
        'test_policy':'one new image-price event after artifact lock; historical test already exposed in other tasks',
        'prior_test_exposed':True,'external_evaluation':'لم يختبر بعد',
        'domain_rule':'parametric PCA residual and distance; threshold val 95th percentile, price targets unused; not validated OOD detector',
        'candidates':'frozen ImageNet features: ridge, SVR, extra trees, histogram boosting; then layer4 log-price fine-tuning',
        'source_hashes':{p.name:sha256(p) for p in [Path(__file__), ROOT/'scripts/image_price_runtime.py']}}
    write_json(run/'protocol.json',protocol)
    print('RUN='+str(run),flush=True); print('ELIGIBLE',protocol['splits'],'EXCLUDED',len(eligible)-len(rows),flush=True)
    return run


def validate_rows(rows):
    if len({r['listing_uid'] for r in rows}) != len(rows): raise ValueError('أكثر من صورة لكل إعلان')
    groups = defaultdict(set); hashes = defaultdict(set)
    for r in rows:
        groups[r['group_id']].add(r['split']); hashes[r['sha256']].add(r['split'])
    if any(len(v)>1 for v in groups.values()) or any(len(v)>1 for v in hashes.values()):
        raise ValueError('تسرب مجموعة أو SHA بين الأقسام')


def data(run, split):
    rows = [r for r in read_csv(run/'selected_images.csv') if r['split']==split]
    listings = {r['listing_uid']:r for r in read_csv(PROCESSED/'listings.csv')}
    y = np.asarray([float(listings[r['listing_uid']]['sar_price']) for r in rows])
    if not np.all(np.isfinite(y) & (y>0)): raise ValueError('هدف سعر غير صالح')
    return rows, y, [listings[r['listing_uid']] for r in rows]


def features(rows):
    index = {r['sha256']:i for i,r in enumerate(read_csv(EMBED/'image_index.csv'))}
    array = np.load(EMBED/'embeddings.npy', mmap_mode='r')
    # مصفوفة ميزات ImageNet غير مدربة على المشروع؛ لا تقرأ الصفوف الأخرى للتجربة.
    return np.array(array[[index[r['sha256']] for r in rows]])


def regression_metrics(rows,y,p,bootstrap=True):
    m=metrics(y,p); m['MAPE_percent']=float(np.mean(np.abs(y-p)/y)*100)
    if bootstrap:
        groups=sorted({r['group_id'] for r in rows}); lookup={g:i for i,g in enumerate(groups)}
        stats=np.zeros((len(groups),3))
        for r,a,b in zip(rows,y,p): stats[lookup[r['group_id']]] += [1,abs(a-b),abs(a-b)/a*100]
        w=np.random.default_rng(SEED).multinomial(len(groups),np.full(len(groups),1/len(groups)),size=2000)
        boot=w@stats
        m['ci95']={'MAE_SAR':np.quantile(boot[:,1]/boot[:,0],[.025,.975]).tolist(),
                   'MAPE_percent':np.quantile(boot[:,2]/boot[:,0],[.025,.975]).tolist(),
                   'unit':'group_id','groups':len(groups),'resamples':2000,'conditional_on_selected_model':True}
    return m


def paired_error_ci(rows,y,p,reference):
    groups=sorted({r['group_id'] for r in rows});lookup={g:i for i,g in enumerate(groups)};s=np.zeros((len(groups),2))
    for r,a,b,c in zip(rows,y,p,reference):s[lookup[r['group_id']]] += [1,abs(a-b)-abs(a-c)]
    w=np.random.default_rng(SEED).multinomial(len(groups),np.full(len(groups),1/len(groups)),size=2000)
    boot=w@s
    return {'difference':'image_only_MAE_minus_reference_MAE','ci95':np.quantile(boot[:,1]/boot[:,0],[.025,.975]).tolist(),
            'negative_favors_image_only':True,'unit':'group_id'}


def predict_regressor(model,x,bounds):
    return np.exp(np.clip(model.predict(x),*bounds))


def selection_allowed(run):
    if (run/'selection_lock.json').exists(): raise ValueError('اختيارات النموذج مقفلة؛ لا تجارب إضافية')


def train(run):
    selection_allowed(run); check_inputs(run)
    tr,ytr,_=data(run,'train'); va,yva,_=data(run,'val'); xtr=features(tr); xva=features(va)
    bounds=[float(np.log(ytr.min())),float(np.log(ytr.max()))]
    write_json(run/'target_bounds.json',bounds)
    counts=Counter(r['group_id'] for r in tr)
    weights=np.asarray([1/counts[r['group_id']] for r in tr]); weights/=weights.mean()
    candidates=[]
    for alpha in [.1,1.,10.,100.]: candidates.append((f'ridge_{alpha}',Ridge(alpha=alpha)))
    for c in [1.,10.,100.]: candidates.append((f'svr_{c}',SVR(C=c,epsilon=.05,kernel='rbf',gamma='scale')))
    for leaf in [3,10]: candidates.append((f'extra_{leaf}',ExtraTreesRegressor(n_estimators=300,min_samples_leaf=leaf,max_features=.8,n_jobs=4,random_state=SEED)))
    candidates.append(('hist_15',HistGradientBoostingRegressor(loss='absolute_error',max_iter=250,max_leaf_nodes=15,min_samples_leaf=15,early_stopping=False,learning_rate=.04,random_state=SEED)))
    results=[]; best=float('inf'); selected=None; start=time.perf_counter(); best_model=None
    identity=load_json(run/'protocol.json')['input_check']
    for name,model in candidates:
        t=time.perf_counter(); model.fit(xtr,np.log(ytr),sample_weight=weights)
        p=predict_regressor(model,xva,bounds); m=regression_metrics(va,yva,p,False)
        accepted=m['MAE_SAR']<best
        results.append({'candidate':name,'val_MAE_SAR':m['MAE_SAR'],'val_MAPE_percent':m['MAPE_percent'],
                        'accepted':accepted,'seconds':time.perf_counter()-t,'kind':'frozen',
                        'listings_sha256':identity['files']['listings.csv']['sha256'],
                        'images_sha256':identity['files']['images.csv']['sha256'],
                        'split_fingerprint':identity['split_fingerprint']})
        if accepted:
            best=m['MAE_SAR']; selected={'name':name,'kind':'frozen','val_MAE_SAR':best}
            best_model=model
            joblib.dump(model,run/(name+'.joblib'),compress=3)
        if name=='ridge_1.0': joblib.dump(model,run/'ridge_initializer.joblib',compress=3)
        write_csv(run/'experiments.csv',results); write_json(run/'selected.json',selected)
        print(name,'val MAE',round(m['MAE_SAR'],2),'KEEP' if accepted else 'REJECT',flush=True)
    joblib.dump(best_model,run/'regressor.joblib',compress=3)
    # تشخيص GroupKFold من train وحده؛ لا تغيير للاختيار وفق نتيجة test.
    g=np.asarray([r['group_id'] for r in tr]);oof=np.zeros(len(tr))
    for fit_ix,hold_ix in GroupKFold(n_splits=3).split(xtr,groups=g):
        if set(g[fit_ix]) & set(g[hold_ix]):raise ValueError('تسرب مجموعة في CV')
        model=clone(best_model).fit(xtr[fit_ix],np.log(ytr[fit_ix]),sample_weight=weights[fit_ix])
        fold_bounds=[float(np.log(ytr[fit_ix].min())),float(np.log(ytr[fit_ix].max()))]
        oof[hold_ix]=predict_regressor(model,xtr[hold_ix],fold_bounds)
    write_json(run/'train_group_cv.json',{'candidate':selected['name'],'folds':3,
        'out_of_fold':regression_metrics(tr,ytr,oof),
        'in_sample':regression_metrics(tr,ytr,predict_regressor(best_model,xtr,bounds),False)})
    # فحص نطاق بارامتري؛ يخزن معاملات توزيع، لا صور/أسعار/فهرس بحث.
    pca=PCA(n_components=64,svd_solver='full').fit(xtr)
    z=pca.transform(xtr); residual=np.mean((xtr-pca.inverse_transform(z))**2,axis=1)
    distance=np.mean(z*z/np.maximum(pca.explained_variance_,1e-8),axis=1)
    detector={'pca':pca,'residual_scale':float(max(np.quantile(residual,.95),1e-12)),
              'distance_scale':float(max(np.quantile(distance,.95),1e-12))}
    threshold=float(np.quantile(domain_score(detector,xva),.95))
    joblib.dump(detector,run/'domain.joblib',compress=3)
    write_json(run/'domain_policy.json',{'threshold':threshold,'selection':'95th percentile of val feature distribution; target-free',
        'calibrated_probability':False,'unknown_car_detection_validated':False})
    write_json(run/'train_timing.json',{'frozen_seconds':time.perf_counter()-start})
    check_inputs(run)


def make_net(run):
    net=resnet18(weights=None)
    net.load_state_dict(torch.load(EMBED/'encoder_weights.pth',map_location='cpu',weights_only=True))
    net.fc=torch.nn.Linear(512,1)
    return net


def forward_tail(net,z):
    # نفس تطبيع المتجه إلى طول واحد المستخدم في مشغل الصور الخام.
    f=net.avgpool(net.layer4(z)).flatten(1)
    return net.fc(torch.nn.functional.normalize(f,dim=1)).flatten()


def finetune(run):
    selection_allowed(run); check_inputs(run); torch.manual_seed(SEED)
    tr,ytr,_=data(run,'train');va,yva,_=data(run,'val')
    index={r['sha256']:i for i,r in enumerate(read_csv(ACT/'index.csv'))}
    arrays=[np.load(ACT/f'stage3_{v}.npy',mmap_mode='r') for v in [0,1]]
    trids=np.array([index[r['sha256']] for r in tr]); vaids=np.array([index[r['sha256']] for r in va])
    bounds=load_json(run/'target_bounds.json'); selected=load_json(run/'selected.json'); best=selected['val_MAE_SAR']
    results=read_csv(run/'experiments.csv'); initial=joblib.load(run/'ridge_initializer.joblib')
    counts=Counter(r['group_id'] for r in tr); w=np.array([1/counts[r['group_id']] for r in tr]);w/=w.mean()
    for lr in [1e-5,3e-5]:
        torch.manual_seed(SEED);rng=np.random.default_rng(SEED); net=make_net(run)
        with torch.no_grad():
            net.fc.weight.copy_(torch.tensor(initial.coef_[None],dtype=torch.float32))
            net.fc.bias.copy_(torch.tensor([initial.intercept_],dtype=torch.float32))
        for p in net.parameters():p.requires_grad=False
        for p in list(net.layer4.parameters())+list(net.fc.parameters()):p.requires_grad=True
        optimizer=torch.optim.AdamW([p for p in net.parameters() if p.requires_grad],lr=lr,weight_decay=.001)
        patience=0;local_best=float('inf')
        for epoch in range(1,7):
            net.eval() # إحصاءات BatchNorm تبقى ImageNet؛ gradients لطبقة 4 والرأس فقط.
            start=time.perf_counter(); order=rng.permutation(len(tr)); losses=[]
            for first in range(0,len(tr),32):
                ix=order[first:first+32]; flip=int(rng.integers(2)); z=torch.from_numpy(np.array(arrays[flip][trids[ix]]))
                pred=forward_tail(net,z); target=torch.tensor(np.log(ytr[ix]),dtype=torch.float32)
                loss=(torch.nn.functional.smooth_l1_loss(pred,target,beta=.2,reduction='none')*torch.tensor(w[ix],dtype=torch.float32)).mean()
                optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),1.);optimizer.step();losses.append(float(loss.detach()))
            out=[]
            with torch.inference_mode():
                for first in range(0,len(va),64):
                    out.extend(forward_tail(net,torch.from_numpy(np.array(arrays[0][vaids[first:first+64]]))).tolist())
            p=np.exp(np.clip(out,*bounds)); m=regression_metrics(va,yva,p,False); accepted=m['MAE_SAR']<best
            name=f'layer4_lr{lr}_epoch{epoch}'
            results.append({'candidate':name,'val_MAE_SAR':m['MAE_SAR'],'val_MAPE_percent':m['MAPE_percent'],
                'accepted':accepted,'seconds':time.perf_counter()-start,'kind':'tail',
                'listings_sha256':load_json(run/'protocol.json')['input_check']['files']['listings.csv']['sha256'],
                'images_sha256':load_json(run/'protocol.json')['input_check']['files']['images.csv']['sha256'],
                'split_fingerprint':load_json(run/'protocol.json')['input_check']['split_fingerprint']})
            if accepted:
                best=m['MAE_SAR'];selected={'name':name,'kind':'tail','val_MAE_SAR':best,'normalized_features':True,
                                          'checkpoint':name+'.pt'}
                torch.save(net.state_dict(),run/selected['checkpoint'])
            if m['MAE_SAR']<local_best:local_best=m['MAE_SAR'];patience=0
            else:patience+=1
            write_csv(run/'experiments.csv',results);write_json(run/'selected.json',selected)
            print(name,'val MAE',round(m['MAE_SAR'],2),'KEEP' if accepted else 'REJECT','seconds',round(time.perf_counter()-start,1),flush=True)
            if patience>=2:break
    check_inputs(run)


def tail_predictions(run,rows):
    net=make_net(run);net.load_state_dict(torch.load(run/load_json(run/'selected.json')['checkpoint'],map_location='cpu',weights_only=True));net.eval()
    tf=ResNet18_Weights.IMAGENET1K_V1.transforms(); output=[]
    with torch.inference_mode():
        for first in range(0,len(rows),32):
            batch=rows[first:first+32];xs=[]
            for r in batch:
                path=ROOT/'الملفات_المهمة'/r['relative_path']
                if sha256(path)!=r['sha256']:raise ValueError('تغيرت صورة')
                xs.append(tf(load_image(path)))
            x=torch.stack(xs);z=net.maxpool(net.relu(net.bn1(net.conv1(x))));z=net.layer3(net.layer2(net.layer1(z)))
            output.extend(forward_tail(net,z).tolist())
    return np.exp(np.clip(output,*load_json(run/'target_bounds.json')))


def refine(run):
    selection_allowed(run);check_inputs(run)
    chosen=load_json(run/'selected.json')
    if chosen['kind']!='tail':return
    start_checkpoint=chosen['checkpoint'];best=chosen['val_MAE_SAR']
    tr,ytr,_=data(run,'train');va,yva,_=data(run,'val');bounds=load_json(run/'target_bounds.json')
    index={r['sha256']:i for i,r in enumerate(read_csv(ACT/'index.csv'))}
    arrays=[np.load(ACT/f'stage3_{v}.npy',mmap_mode='r') for v in [0,1]]
    train_ids=np.array([index[r['sha256']] for r in tr]);val_ids=np.array([index[r['sha256']] for r in va])
    counts=Counter(r['group_id'] for r in tr);w=np.array([1/counts[r['group_id']] for r in tr]);w/=w.mean()
    identity=load_json(run/'protocol.json')['input_check'];results=read_csv(run/'experiments.csv')
    def evaluate(net,ids):
        out=[]
        with torch.inference_mode():
            for first in range(0,len(ids),64):
                out.extend(forward_tail(net,torch.from_numpy(np.array(arrays[0][ids[first:first+64]]))).tolist())
        return np.exp(np.clip(out,*bounds))
    for loss_kind in ['log','sar']:
        torch.manual_seed(SEED);rng=np.random.default_rng(SEED)
        net=make_net(run);net.load_state_dict(torch.load(run/start_checkpoint,map_location='cpu',weights_only=True));net.eval()
        for p in net.parameters():p.requires_grad=False
        for p in list(net.layer4.parameters())+list(net.fc.parameters()):p.requires_grad=True
        optim=torch.optim.AdamW([p for p in net.parameters() if p.requires_grad],lr=1e-5,weight_decay=.001)
        local_best=chosen['val_MAE_SAR'];patience=0
        for epoch in range(1,7):
            t=time.perf_counter();order=rng.permutation(len(tr))
            for first in range(0,len(tr),32):
                ix=order[first:first+32];z=torch.from_numpy(np.array(arrays[int(rng.integers(2))][train_ids[ix]]))
                pred=forward_tail(net,z);y=torch.tensor(ytr[ix],dtype=torch.float32)
                if loss_kind=='log':error=torch.nn.functional.smooth_l1_loss(pred,torch.log(y),beta=.2,reduction='none')
                else:error=torch.abs(torch.exp(torch.clamp(pred,*bounds))-y)/float(np.median(ytr))
                loss=(error*torch.tensor(w[ix],dtype=torch.float32)).mean()
                optim.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),1.);optim.step()
            p=evaluate(net,val_ids);m=regression_metrics(va,yva,p,False)
            train_mae=float(np.mean(abs(ytr-evaluate(net,train_ids))))
            accepted=m['MAE_SAR']<best;name=f'refine_{loss_kind}_epoch{epoch}'
            results.append({'candidate':name,'val_MAE_SAR':m['MAE_SAR'],'val_MAPE_percent':m['MAPE_percent'],
                'accepted':accepted,'seconds':time.perf_counter()-t,'kind':'tail',
                'listings_sha256':identity['files']['listings.csv']['sha256'],'images_sha256':identity['files']['images.csv']['sha256'],
                'split_fingerprint':identity['split_fingerprint'],'train_MAE_SAR':train_mae})
            if accepted:
                best=m['MAE_SAR'];chosen_new={'name':name,'kind':'tail','val_MAE_SAR':best,'normalized_features':True,
                                            'checkpoint':name+'.pt','start_checkpoint':start_checkpoint}
                torch.save(net.state_dict(),run/chosen_new['checkpoint']);write_json(run/'selected.json',chosen_new)
            write_csv(run/'experiments.csv',results)
            print(name,'train MAE',round(train_mae,1),'val MAE',round(m['MAE_SAR'],1),'KEEP' if accepted else 'REJECT',flush=True)
            if m['MAE_SAR']<local_best:local_best=m['MAE_SAR'];patience=0
            else:patience+=1
            if patience>=2:break
    check_inputs(run)


def predictions(run,rows):
    if load_json(run/'selected.json')['kind']=='tail':return tail_predictions(run,rows)
    return predict_regressor(joblib.load(run/'regressor.joblib'),features(rows),load_json(run/'target_bounds.json'))


def verify_cache(run):
    check_inputs(run)
    rows=[r for r in read_csv(run/'selected_images.csv') if r['split'] in {'train','val'}]
    ix={r['sha256']:i for i,r in enumerate(read_csv(ACT/'index.csv'))}
    a=[np.load(ACT/f'stage3_{v}.npy',mmap_mode='r') for v in [0,1]]
    net=resnet18(weights=None);net.load_state_dict(torch.load(EMBED/'encoder_weights.pth',map_location='cpu',weights_only=True));net.eval()
    tf=ResNet18_Weights.IMAGENET1K_V1.transforms();largest=0.;start=time.perf_counter()
    with torch.inference_mode():
        for first in range(0,len(rows),32):
            batch=rows[first:first+32];xs=[]
            for r in batch:
                path=ROOT/'الملفات_المهمة'/r['relative_path']
                if sha256(path)!=r['sha256']:raise ValueError('تغيرت صورة خام')
                xs.append(tf(load_image(path)))
            x=torch.stack(xs);ids=[ix[r['sha256']] for r in batch]
            for flip in [0,1]:
                z=x.flip(-1) if flip else x
                z=net.maxpool(net.relu(net.bn1(net.conv1(z))))
                z=net.layer3(net.layer2(net.layer1(z)))
                largest=max(largest,float(np.max(np.abs(z.numpy()-np.array(a[flip][ids])))))
            if first%512==0:print('VERIFY CACHE',first+len(batch),'/',len(rows),flush=True)
    result={'verified_images':len(rows),'views':2,'max_activation_difference':largest,'seconds':time.perf_counter()-start,
            'test_images_used':False,'passed':largest<1e-4}
    write_json(run/'cache_verification.json',result)
    if not result['passed']:raise ValueError('كاش التدريب لا يطابق الصور الأصلية')
    print('CACHE',result,flush=True)


def comparison(run,split):
    rows,y,listing_rows=data(run,split);tr,ytr,trl=data(run,'train');p=predictions(run,rows)
    baseline=np.full(len(rows),np.median(ytr))
    # مرجع خصائص مكتوبة على نفس الإعلانات؛ لا يدخل في النموذج الصوري أو الاختيار.
    original=ROOT/load_json(ROOT/'models/price_current.json')['run_dir']
    if split=='test':
        candidate=load_json(original/'metadata.json')['selected_candidate']
        saved={r['listing_uid']:r for r in read_csv(original/'test_predictions.csv') if r['candidate']==candidate}
        if any(r['listing_uid'] not in saved for r in rows):raise ValueError('تنبؤ مرجع محفوظ مفقود')
        tabular=np.array([float(saved[r['listing_uid']]['predicted_sar']) for r in rows])
        if any(float(saved[r['listing_uid']]['actual_sar'])!=a for r,a in zip(rows,y)):raise ValueError('هدف المرجع المحفوظ مختلف')
    else:
        tabular=joblib.load(original/'model.joblib')['model'].predict(listing_rows)
    detector=joblib.load(run/'domain.joblib'); score=domain_score(detector,features(rows))
    accepted=score<=load_json(run/'domain_policy.json')['threshold']
    out={'image_only':regression_metrics(rows,y,p),'train_median':regression_metrics(rows,y,baseline),
         'existing_tabular_reference':regression_metrics(rows,y,tabular),
         'paired_vs_median':paired_error_ci(rows,y,p,baseline),'paired_vs_tabular':paired_error_ci(rows,y,p,tabular),
         'selective':{'coverage':float(accepted.mean()),'n':int(accepted.sum()),
                      'metrics':regression_metrics([r for r,k in zip(rows,accepted) if k],y[accepted],p[accepted]) if accepted.any() else None}}
    write_json(run/f'{split}_metrics.json',out)
    write_csv(run/f'{split}_predictions.csv',[{**r,'actual_sar':float(a),'predicted_sar':float(b),'baseline_sar':float(d),
        'tabular_reference_sar':float(t),'absolute_error_sar':float(abs(a-b)),'domain_score':float(s),'accepted':bool(k)}
        for r,a,b,d,t,s,k in zip(rows,y,p,baseline,tabular,score,accepted)])
    segments=[]
    for make in sorted({r['make'] for r in listing_rows}):
        ix=[i for i,r in enumerate(listing_rows) if r['make']==make]
        segments.append({'make':make,**regression_metrics([rows[i] for i in ix],y[ix],p[ix],False)})
    write_csv(run/f'{split}_by_make.csv',segments)
    bands=[]
    for label,low,high in [('below_50000',0,50000),('50000_to_150000',50000,150000),('150000_and_over',150000,float('inf'))]:
        ix=np.flatnonzero((y>=low)&(y<high))
        if len(ix):bands.append({'price_band':label,**regression_metrics([rows[i] for i in ix],y[ix],p[ix],False)})
    write_csv(run/f'{split}_by_price_band.csv',bands)
    return out


def lock(run):
    selection_allowed(run); check_inputs(run)
    if not load_json(run/'cache_verification.json',{}).get('passed'):raise ValueError('يجب التحقق من الكاش قبل القفل')
    result=comparison(run,'val');selected=load_json(run/'selected.json')
    if not np.isclose(result['image_only']['MAE_SAR'],selected['val_MAE_SAR'],atol=.1,rtol=1e-5):
        raise ValueError('تنبؤ الصور الخام لا يطابق نتيجة التحقق أثناء التدريب')
    shutil.copy2(EMBED/'encoder_weights.pth',run/'encoder_weights.pth')
    shutil.copy2(ROOT/'scripts/image_price_runtime.py',run/'image_price_runtime.py')
    names=['encoder_weights.pth','domain.joblib']
    if selected['kind']=='tail':
        shutil.copy2(run/selected['checkpoint'],run/'price_network.pt');names.append('price_network.pt')
    else:names.append('regressor.joblib')
    policy={'kind':selected['kind'],'feature_normalization':'L2','log_prediction_bounds':load_json(run/'target_bounds.json'),
        'domain_threshold':load_json(run/'domain_policy.json')['threshold'],'status':'experimental',
        'artifacts_sha256':{n:sha256(run/n) for n in names}}
    write_json(run/'runtime_policy.json',policy)
    write_json(run/'selection_lock.json',{'locked_at':now(),'selected':selected,'test_not_yet_evaluated':True,
        'final_code_sha256':{p.name:sha256(p) for p in [Path(__file__),ROOT/'scripts/image_price_runtime.py']},
        'sha256':{n:sha256(run/n) for n in names+['runtime_policy.json','selected.json','image_price_runtime.py']}})
    print('LOCKED',selected,'VAL',result['image_only'],flush=True)


def test_once(run):
    check_inputs(run);lock_data=load_json(run/'selection_lock.json')
    if not lock_data:raise ValueError('يجب قفل الاختيار قبل test')
    for n,h in lock_data['sha256'].items():
        if sha256(run/n)!=h:raise ValueError('تغير نموذج بعد القفل')
    with (run/'test_once.json').open('x',encoding='utf-8') as f:
        json.dump({'started_at':now(),'count':1,'task':'new_image_only_price','prior_test_exposed':True,
                   'evaluation_script_sha256':sha256(Path(__file__)),
                   'tabular_reference':'reuse saved original test predictions; no additional forward evaluation'},f)
    result=comparison(run,'test')
    write_json(run/'test_completed.json',{'completed_at':now(),'count':1})
    check_inputs(run);print('TEST',json.dumps(result,ensure_ascii=False),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['prepare','train','finetune','refine','verify-cache','lock','test']);parser.add_argument('--run',type=Path)
    args=parser.parse_args()
    if args.command=='prepare':prepare()
    else:
        if args.run is None:parser.error('--run is required')
        {'train':train,'finetune':finetune,'refine':refine,'verify-cache':verify_cache,'lock':lock,'test':test_once}[args.command](args.run.resolve())
