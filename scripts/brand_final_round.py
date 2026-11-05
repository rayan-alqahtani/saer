"""جولة نهائية للماركة: اختيار داخل train بالمجموعات، ثم قياس val دون إعادة test."""
import argparse
import copy
import csv
import json
import time
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import torch
from torch import nn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import classification_report, confusion_matrix
from PIL import Image, ImageOps
from torchvision.models import ResNet18_Weights
from improve_common import (ROOT, PROCESSED, guard, session, compact_rows, read_csv,
                            load_json, save_json, save_csv, sha256, score, class_bootstrap)
from improve_visual import load_base, activation_inputs
from ml_visual import embedding_data, select_rejection_threshold

torch.set_num_threads(4)
SEED = 20261022
MAX_EPOCHS = 6


class BrandNet(nn.Module):
    def __init__(self, nclasses):
        super().__init__()
        self.backbone = load_base()
        self.backbone.fc = nn.Identity()
        self.head = nn.Linear(512, nclasses)

    def forward(self, x, stage=0):
        b = self.backbone
        if stage == 0:
            x = b.maxpool(b.relu(b.bn1(b.conv1(x))))
            x = b.layer2(b.layer1(x))
        if stage in (0, 2):
            x = b.layer3(x)
        x = b.layer4(x)
        return self.head(nn.functional.normalize(b.avgpool(x).flatten(1), dim=1))


def strict_inputs():
    check = guard()
    if not all(f['authorized_match'] for f in check['files'].values()):
        raise ValueError('بصمة البيانات المعتمدة تغيرت')
    return check


def internal_units(rows, run):
    """تجميع احترازي داخلي للصور المتقاربة جدًا؛ لا يكتب group_id أو split في البيانات."""
    original = {r['image_uid']:r for r in read_csv(PROCESSED/'images.csv')}
    parent = {r['group_id']:r['group_id'] for r in rows}
    def find(g):
        while parent[g] != g:
            parent[g] = parent[parent[g]]
            g = parent[g]
        return g
    ph = [int(original[r['image_uid']]['phash'],16) for r in rows]
    dh = [int(original[r['image_uid']]['dhash'],16) for r in rows]
    thumbs = {}
    def thumb(i):
        if i not in thumbs:
            with Image.open(ROOT/'الملفات_المهمة'/rows[i]['relative_path']) as im:
                thumbs[i] = np.asarray(ImageOps.exif_transpose(im).convert('RGB').resize((64,64)),dtype=np.float32)
        return thumbs[i]
    joins = []
    for i,r in enumerate(rows):
        for j in range(i+1,len(rows)):
            if r['group_id']==rows[j]['group_id']:continue
            if (ph[i]^ph[j]).bit_count()>4 or (dh[i]^dh[j]).bit_count()>4:continue
            mae=float(np.abs(thumb(i)-thumb(j)).mean())
            if mae<=3:
                a,b=find(r['group_id']),find(rows[j]['group_id'])
                parent[a]=b
                joins.append({'left':r['image_uid'],'right':rows[j]['image_uid'],'mae':mae})
    save_json(run/'inner_near_clusters.json',{'rule':'pHash<=4,dHash<=4,RGB64 MAE<=3; conservative grouping, not duplicate proof','pairs':joins})
    print('INNER_NEAR_GROUPS',len(joins),flush=True)
    return np.asarray([find(r['group_id']) for r in rows])


def weights_for(rows, balance):
    groups = Counter(r['group_id'] for r in rows)
    support = Counter()
    seen = set()
    for r in rows:
        key = r['group_id'], r['label_make']
        if key not in seen:
            support[r['label_make']] += 1
            seen.add(key)
    w = np.asarray([1/groups[r['group_id']] / (support[r['label_make']]**0.5 if balance else 1)
                    for r in rows])
    return w / w.mean()


def initialize(rows, classes, balance):
    torch.manual_seed(SEED)
    model = BrandNet(len(classes))
    _, _, lookup, vectors, _ = embedding_data()
    x = np.asarray(vectors[[lookup[r['sha256']] for r in rows]])
    labels = [r['label_make'] for r in rows]
    clf = LogisticRegression(C=10., max_iter=600).fit(x, labels, sample_weight=weights_for(rows, balance))
    assert list(clf.classes_) == classes
    with torch.no_grad():
        model.head.weight.copy_(torch.as_tensor(clf.coef_, dtype=torch.float32))
        model.head.bias.copy_(torch.as_tensor(clf.intercept_, dtype=torch.float32))
    return model


def evaluate(model, rows, tta=False):
    array, ids = activation_inputs(rows, 2)
    flip, _ = activation_inputs(rows, 2, 1)
    out = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(rows), 48):
            pos = ids[start:start+48]
            p = model(torch.from_numpy(np.array(array[pos])), 2).softmax(1)
            if tta:
                p = (p + model(torch.from_numpy(np.array(flip[pos])), 2).softmax(1))/2
            out.append(p.numpy())
    return np.concatenate(out)


def fit(rows, classes, cfg, folder, tune=None, fixed_epochs=None):
    model = initialize(rows, classes, cfg['balance'])
    for p in model.parameters():
        p.requires_grad = False
    blocks = [model.backbone.layer4]
    if cfg['depth'] == 3:
        blocks.append(model.backbone.layer3)
    for block in blocks + [model.head]:
        for p in block.parameters():
            p.requires_grad = True
    optimizer = torch.optim.AdamW([
        {'params': [p for b in blocks for p in b.parameters()], 'lr': 3e-5},
        {'params': model.head.parameters(), 'lr': 1e-3}], weight_decay=.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS)
    stage = 3 if cfg['depth'] == 4 else 2
    array, ids = activation_inputs(rows, stage)
    flip, _ = activation_inputs(rows, stage, 1)
    targets = np.asarray([classes.index(r['label_make']) for r in rows])
    weights = weights_for(rows, cfg['balance']).astype('float32')
    rng = np.random.default_rng(SEED)
    best, beststate, bestepoch, bad = -1., None, 0, 0
    if tune is not None:
        baseline = score([r['label_make'] for r in tune],evaluate(model,tune),classes)
        best, beststate = baseline['macro_F1'],copy.deepcopy(model.state_dict())
        save_json(folder/'initial_head_metrics.json',baseline)
    history = []
    for epoch in range(MAX_EPOCHS if fixed_epochs is None else fixed_epochs):
        start = time.monotonic()
        model.train()
        for module in model.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()
        order = rng.permutation(len(rows))
        losses = []
        for first in range(0, len(rows), 32):
            batch = order[first:first+32]
            pos = ids[batch]
            x = np.array(array[pos])
            mask = rng.random(len(batch)) < .5
            x[mask] = flip[pos[mask]]
            optimizer.zero_grad(set_to_none=True)
            logits = model(torch.from_numpy(x), stage)
            loss = (nn.functional.cross_entropy(logits, torch.as_tensor(targets[batch]), reduction='none')
                    * torch.from_numpy(weights[batch])).mean()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step()
            losses.append(loss.item())
            if first % 2048 == 0:
                print(folder.name, 'epoch', epoch+1, first, '/', len(rows), flush=True)
        scheduler.step()
        record = {'epoch': epoch+1, 'weighted_train_loss': float(np.mean(losses)),
                  'seconds': time.monotonic()-start, **cfg}
        if tune is not None:
            probs = evaluate(model, tune)
            record['inner_metrics'] = score([r['label_make'] for r in tune], probs, classes)
            f1 = record['inner_metrics']['macro_F1']
            if f1 > best + 1e-4:
                best, beststate, bestepoch, bad = f1, copy.deepcopy(model.state_dict()), epoch+1, 0
            else:
                bad += 1
        history.append(record)
        save_json(folder/'history.json', history)
        print('EPOCH_RESULT', folder.name, json.dumps(record), flush=True)
        if tune is not None and bad >= 2:
            break
    if tune is not None:
        model.load_state_dict(beststate)
    return model, bestepoch, history


def main():
    run = Path((ROOT/'tmp/brand_final_run.txt').read_text().strip())
    if (run/'selection.json').exists():
        raise ValueError('الجولة مقفلة؛ لا تدريب ثان بعد رؤية val')
    check = strict_inputs()
    tr = compact_rows('make', 'train')
    # لا نستخدم رؤوسًا أو شبكات ضُبطت سابقًا على كل train أو val داخل تقسيم الاختيار الداخلي.
    classes = sorted({r['label_make'] for r in tr})
    groups = internal_units(tr,run)
    y = np.asarray([r['label_make'] for r in tr])
    folds = list(StratifiedGroupKFold(5, shuffle=True, random_state=SEED).split(np.zeros(len(y)), y, groups))
    fit_ids, tune_ids = next((a, b) for a, b in folds if set(y[a]) == set(classes) and set(y[b]) == set(classes))
    train = [tr[i] for i in fit_ids]
    tune = [tr[i] for i in tune_ids]
    assert not (set(groups[fit_ids]) & set(groups[tune_ids]))
    # كل أصول train وval تُطابق بصمتها قبل استعمال كاش الصور.
    for r in tr + compact_rows('make', 'val'):
        if sha256(ROOT/'الملفات_المهمة'/r['relative_path']) != r['sha256']:
            raise ValueError('تغيرت صورة: '+r['image_uid'])
    save_csv(run/'inner_partition.csv', [{'image_uid':r['image_uid'], 'group_id':r['group_id'],
             'role':role} for role, rows in [('fit',train),('tune',tune)] for r in rows])
    save_json(run/'protocol.json', {'classes':classes, 'inputs':check, 'seed':SEED,
        'selection':'train-only StratifiedGroupKFold; original val final comparison only',
        'init':'ImageNet backbone and train-partition-only LogisticRegression head',
        'max_epochs':MAX_EPOCHS, 'early_stopping_patience':2,
        'test_predictions':0, 'configs':[{'depth':4,'balance':False}, {'depth':4,'balance':True}, {'depth':3,'balance':True}],
        'source_sha256':sha256(Path(__file__)), 'prior_val_exposed':True})
    candidates = []
    for cfg in [{'depth':4,'balance':False}, {'depth':4,'balance':True}, {'depth':3,'balance':True}]:
        strict_inputs()
        folder = run/f"inner_depth{cfg['depth']}_balanced{cfg['balance']}"
        folder.mkdir(exist_ok=False)
        net, epoch, history = fit(train, classes, cfg, folder, tune=tune)
        for tta in [False, True]:
            probs = evaluate(net, tune, tta)
            result = {**cfg, 'epochs':epoch, 'tta':tta, **score([r['label_make'] for r in tune], probs, classes)}
            candidates.append(result)
            print('INNER_CANDIDATE', json.dumps(result), flush=True)
        save_csv(run/'inner_experiments.csv', candidates)
        torch.save({'state_dict':net.state_dict(), 'classes':classes},folder/'inner_model.pt')
    chosen = max(candidates, key=lambda r:r['macro_F1'])
    save_json(run/'selection.json', {'chosen_on_train_only':chosen,'test_evaluated':False,'original_val_used':False})
    folder = run/'refit_all_train';folder.mkdir()
    net, _, _ = fit(tr, classes, chosen, folder, fixed_epochs=chosen['epochs'])
    checkpoint = run/'brand_model.pt'
    torch.save({'state_dict':net.state_dict(), 'classes':classes, 'tta':chosen['tta']}, checkpoint)
    save_json(run/'final_lock.json', {'checkpoint_sha256':sha256(checkpoint), 'selection_sha256':sha256(run/'selection.json')})
    # الحدث الوحيد في هذه الجولة لقياس val؛ test غير مستدعى.
    with (run/'val_once.json').open('x',encoding='utf-8') as f:
        json.dump({'state':'started','checkpoint_sha256':sha256(checkpoint)},f)
    va = compact_rows('make','val')
    probs = evaluate(net,va,chosen['tta'])
    np.save(run/'val_probabilities.npy', probs)
    labels = np.asarray([r['label_make'] for r in va]);pred = np.asarray(classes)[probs.argmax(1)]
    oldmap = {r['image_uid']:r['predicted'] for r in read_csv(session()/'make/val_predictions.csv')}
    oldpred = np.asarray([oldmap[r['image_uid']] for r in va])
    # عزل إعلان الصورة المعاد تحريرها بين val/test عن المقارنة الأساسية فقط؛ الجدول الأصلي ثابت.
    clean = np.asarray([r['listing_uid'] != 'external_group_b:0035US' for r in va])
    result = {'full_val':score(labels,probs,classes),'test_evaluated':False,'selection':chosen}
    for name,mask in [('all_val',np.ones(len(va),dtype=bool)),('audited_val',clean)]:
        rows = [r for r,k in zip(va,mask) if k]
        ci, acc, f1 = class_bootstrap(rows,labels[mask],pred[mask],classes)
        oldci, oa, of = class_bootstrap(rows,labels[mask],oldpred[mask],classes)
        result[name] = {**score(labels[mask],probs[mask],classes),'ci95':ci,
            'baseline_accuracy':float((oldpred[mask]==labels[mask]).mean()),
            'paired_f1_delta_ci95':np.quantile(f1-of,[.025,.975]).tolist()}
    per = classification_report(labels,pred,labels=classes,output_dict=True,zero_division=0)
    save_csv(run/'val_per_class.csv',[{'class':cl,**per[cl]} for cl in classes])
    save_json(run/'val_confusion.json',{'classes':classes,'matrix':confusion_matrix(labels,pred,labels=classes).tolist()})
    save_csv(run/'val_predictions.csv',[{'image_uid':r['image_uid'],'group_id':r['group_id'],
             'actual':a,'predicted':b} for r,a,b in zip(va,labels,pred)])
    # تحقق الاستدلال الخام مقابل الكاش على خمس صور val؛ لا اختيار معاملات بعد هذه النقطة.
    tf = ResNet18_Weights.IMAGENET1K_V1.transforms();xs=[]
    for r in va[:5]:
        with Image.open(ROOT/'الملفات_المهمة'/r['relative_path']) as im:xs.append(tf(ImageOps.exif_transpose(im).convert('RGB')))
    x = torch.stack(xs)
    with torch.inference_mode():
        raw = net(x).softmax(1)
        if chosen['tta']:raw=(raw+net(x.flip(-1)).softmax(1))/2
    error=float(np.abs(raw.numpy()-probs[:5]).max())
    result['raw_cache_max_probability_error']=error
    if error>1e-4:raise ValueError('عدم مطابقة الاستدلال الخام للكاش')
    strict_inputs()
    save_json(run/'results.json',result)
    print('FINAL_RESULT',json.dumps({'full_val':result['full_val'],'raw_cache_error':error,'folder':str(run)}),flush=True)


if __name__=='__main__':main()
