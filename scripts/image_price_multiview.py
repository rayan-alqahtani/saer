"""تجربة مضبوطة: تدريب متعدد الصور، واستدلال بصورة واحدة، دون فتح test."""
import argparse
import collections
import importlib.util
import json
import os
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '4')
os.environ.setdefault('MKL_NUM_THREADS', '4')
import numpy as np
from PIL import Image, ImageOps
from sklearn.model_selection import GroupKFold
from sklearn.svm import SVR
from threadpoolctl import threadpool_limits
from image_price_final import ROOT, OLD, DATA, rows, js, csvwrite, digest, check, met

BASE = ROOT / 'models/image_price_final_20261022T222653'
SEED = 20261023
threadpool_limits(4)


def gf_module(run):
    sys.path.insert(0, str(run / 'vendor'))
    import gemini_features
    return gemini_features


def validate(run):
    check()
    identity = js(run / 'identity.json')
    for name, sha in identity['frozen'].items():
        if digest(ROOT / name) != sha:
            raise ValueError('تغير مدخل مقفل: ' + name)
    if digest(run / 'images.csv') != identity['images_sha256']:
        raise ValueError('تغير اختيار الصور')
    if digest(run / 'protocol.json') != identity['protocol_sha256']:
        raise ValueError('تغير بروتوكول التجربة')


def prepare():
    audit = check()
    run = ROOT / 'models' / ('image_price_multiview_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S'))
    run.mkdir(); (run / 'cache').mkdir(); (run / 'vendor').mkdir()
    selected = rows(OLD / 'selected_images.csv')
    covers = {r['listing_uid']: r for r in selected if r['split'] in ('train', 'val')}
    listings = {r['listing_uid']: r for r in rows(ROOT / 'data/processed/listings.csv') if r['split'] != 'test'}
    source = rows(ROOT / 'data/processed/images.csv')
    compact = {r['image_uid']: r for r in rows(ROOT / 'الملفات_المهمة/ربط_الصور.csv')}
    eligible = [r for r in source if r['is_exterior'] == 'True' and r['content_kind'] == 'vehicle_overview'
                and r['usable_visual_search'] == 'True']
    bysha = collections.defaultdict(list)
    for r in eligible: bysha[r['sha256']].append(r)
    for rr in bysha.values():
        if len({r['split'] for r in rr}) > 1:
            raise ValueError('تسرب SHA بين الأقسام')
    # Unsupervised contamination audit only. Test pixels may be decoded into small
    # thumbnails to confirm duplicates, never passed through the pricing model.
    cross = []; train_risk = set(); parents = {}
    def find(g):
        parents.setdefault(g, g)
        if parents[g] != g: parents[g] = find(parents[g])
        return parents[g]
    def union(a, b): parents[find(a)] = find(b)
    ph = [int(r['phash'], 16) for r in eligible]
    dh = [int(r['dhash'], 16) for r in eligible]
    thumbs = {}
    def thumb(r):
        if r['image_uid'] not in thumbs:
            p = ROOT / 'الملفات_المهمة' / compact[r['image_uid']]['relative_path']
            with Image.open(p) as im:
                thumbs[r['image_uid']] = np.asarray(ImageOps.exif_transpose(im).convert('RGB').resize((64, 64)), dtype=np.float32)
        return thumbs[r['image_uid']]
    for i, r in enumerate(eligible):
        if r['listing_uid'] not in covers or r['split'] != 'train': continue
        for j, s in enumerate(eligible):
            if r['group_id'] == s['group_id']: continue
            if s['split'] == 'train' and j <= i: continue
            p = (ph[i] ^ ph[j]).bit_count()
            if p > 8 or (dh[i] ^ dh[j]).bit_count() > 10: continue
            mae = float(np.abs(thumb(r) - thumb(s)).mean())
            if mae > 3: continue
            if s['split'] in ('val', 'test'): train_risk.add(r['group_id'])
            else: union(r['group_id'], s['group_id'])
            verdict = 'strict_rgb_near_duplicate'
            cross.append({'left': r['image_uid'], 'right': s['image_uid'], 'left_split': r['split'],
                          'right_split': s['split'], 'phash_distance': p, 'rgb_mae': mae, 'verdict': verdict})
    out = []; excluded = collections.Counter(); seen = set()
    for r in sorted(source, key=lambda x: x['image_uid']):
        if r['listing_uid'] not in covers: continue
        if r['is_exterior'] != 'True' or r['content_kind'] != 'vehicle_overview' or r['usable_visual_search'] != 'True':
            excluded['not_eligible_exterior_overview'] += 1; continue
        if r['split'] == 'train' and r['group_id'] in train_risk:
            excluded['train_group_quarantined_for_near_duplicate_risk'] += 1; continue
        if r['sha256'] in seen:
            excluded['exact_duplicate_sha'] += 1; continue
        seen.add(r['sha256'])
        listing = listings[r['listing_uid']]
        if listing['usable_price'] != 'True' or listing['condition'] != 'used': raise ValueError('إعلان سعر غير مؤهل')
        if (listing['split'], listing['group_id']) != (r['split'], r['group_id']): raise ValueError('اختلاف قسم/مجموعة')
        rel = compact[r['image_uid']]['relative_path']
        if digest(ROOT / 'الملفات_المهمة' / rel) != r['sha256']: raise ValueError('تغير ملف صورة')
        out.append({'image_uid': r['image_uid'], 'listing_uid': r['listing_uid'], 'split': r['split'],
                    'group_id': r['group_id'], 'cv_group': find(r['group_id']), 'relative_path': rel,
                    'sha256': r['sha256'], 'price_sar': listing['sar_price'],
                    'is_cover': str(r['image_uid'] == covers[r['listing_uid']]['image_uid'])})
    if len([r for r in out if r['split'] == 'val' and r['is_cover'] == 'True']) != 321:
        raise ValueError('تغيرت مجموعة صور التقييم الأساسية')
    csvwrite(run / 'images.csv', out)
    if cross: csvwrite(run / 'near_duplicate_audit.csv', cross)
    counts = collections.Counter(r['split'] for r in out)
    js(run / 'data_audit.json', {'input_check': audit, 'counts_images': dict(counts),
        'counts_listings': {s: len({r['listing_uid'] for r in out if r['split'] == s}) for s in counts},
        'excluded_image_counts': dict(excluded), 'quarantined_train_groups': sorted(train_risk),
        'near_duplicate_pairs': len(cross), 'test_inference_run': False, 'test_price_targets_used': False,
        'test_pixels_decoded_only_for_unsupervised_duplicate_audit': True,
        'near_audit_limit': 'pHash<=8,dHash<=10; RGB64 MAE<=3 confirms strict near duplicates; not exhaustive for different viewpoints or edits.',
        'raw_files_count': sum(1 for p in (ROOT / 'data/raw').rglob('*') if p.is_file()),
        'external_labeled_images': 0, 'external_evaluation': 'لم يختبر بعد'})
    protocol = {'created_utc': datetime.now(timezone.utc).isoformat(), 'seed': SEED,
        'primary': 'single fixed cover image per listing; exact original selection',
        'secondary': 'all eligible val views; equal total weight per listing; no multi-image inference',
        'filter': 'is_exterior=True,content_kind=vehicle_overview,usable_visual_search=True; usable_price used ads only; exact SHA unique',
        'experiment': 'one cover vs all eligible train views; same frozen ResNet18, descriptors, kernel, log-SVR and candidate budget',
        'feature_stats': 'fitted on fold-train covers only, identically for both arms; no heldout statistics',
        'cv': '3 GroupKFold inside original train only, same cv_group folds; confirmed train near duplicate components share folds; original group_id unchanged',
        'sample_weights': 'sum weights per listing=1 (1/n images), no per-image mean normalization; C remains comparable',
        'grid_C': [10.0, 100.0, 1000.0], 'epsilon': 0.05,
        'kernel': {'full_weight': .28, 'components': [.55, .25, .20], 'gamma': 1.5},
        'selection': 'minimum OOF cover MAE inside train, separately per arm; tie lower C',
        'practical_gate': {'min_cv_MAE_reduction_percent': 5.0, 'max_MAPE_increase_pp': .5,
                           'max_hit20_decrease_pp': 1.0, 'max_expensive_band_MAE_increase_percent': 5.0},
        'bootstrap': '2000 paired draws at original group_id; new-minus-control MAE; CI reported, not sole gate; post-selection exploratory',
        'val': 'two selected arms only once plus frozen archived reference; exploratory historically reused; no val tuning',
        'blend': 'fixed historical 25% Gemini log-prediction on val only; no Gemini supervised head used inside CV',
        'old_test': 'no inference, metrics, model selection, or target use; unsupervised thumbnail duplicate audit only',
        'domain': 'retain old unchanged same-encoder heuristic; no threshold tuning without external calibration',
        'DINOv2': 'only considered if controlled ResNet improvement passes; not part of this run',
        'deployment': 'new experimental candidate only; do not change current model pointer without external validation',
        'external': 'separate calibration and final groups, documented exact-image asking prices; not available yet'}
    js(run / 'protocol.json', protocol)
    freeze = [ROOT / 'data/processed' / n for n in js(ROOT / 'data/processed/training_manifest.json')['input_files_sha256']]
    freeze += [ROOT / 'data/processed/training_manifest.json', ROOT / 'الملفات_المهمة/الداتا_المحدثة.csv',
               ROOT / 'الملفات_المهمة/ربط_الصور.csv', OLD / 'selected_images.csv', DATA / 'dataset.csv',
               ROOT / 'models/image_price_current.json', BASE / 'LOCK.json', BASE / 'head.npz', BASE / 'encoder_weights.pth',
               BASE / 'policy.json', BASE / 'visual_stats.npz', BASE / 'val_predictions.csv']
    js(run / 'identity.json', {'frozen': {p.relative_to(ROOT).as_posix(): digest(p) for p in freeze},
        'images_sha256': digest(run / 'images.csv'), 'protocol_sha256': digest(run / 'protocol.json'),
        'split_fingerprint': js(ROOT / 'data/processed/training_manifest.json')['split_fingerprint']})
    shutil.copy2(BASE / 'encoder_weights.pth', run / 'encoder_weights.pth')
    shutil.copy2(BASE / 'vendor/gemini_features.py', run / 'vendor/gemini_features.py')
    js(ROOT / 'models/image_price_multiview_work.json', {'run_dir': run.relative_to(ROOT).as_posix()})
    print('PREPARED', run, 'images', dict(counts), 'quarantined_groups', len(train_risk), 'confirmed_pairs', len(cross), flush=True)


def extract(run):
    validate(run); gf = gf_module(run); d = rows(run / 'images.csv')
    if (run / 'cache/feature_identity.json').exists(): raise ValueError('الميزات موجودة بالفعل')
    stats = np.load(BASE / 'visual_stats.npz')
    dims = {'crop': 25154, 'full': 21504, 'x53': 15170, 'raw': 154, 'domain': 512}
    arrays = {k: np.lib.format.open_memmap(run / 'cache' / (k + '.npy'), mode='w+', dtype='float32', shape=(len(d), n)) for k, n in dims.items()}
    oldrows = rows(DATA / 'dataset.csv'); lookup = {r['sha256']: i for i, r in enumerate(oldrows)}
    frozen = js(BASE / 'cache/identity.json')
    for name, sha in frozen['files'].items():
        if digest(BASE / 'cache' / name) != sha: raise ValueError('مخزن المرجع غير مطابق')
    old = {k: np.load(BASE / 'cache' / (k + '.npy'), mmap_mode='r') for k in ('crop', 'full', 'x53', 'domain')}
    oldraw = np.load(BASE / 'cache/stats.npz')['vs']
    remaining = []
    for i, r in enumerate(d):
        if r['sha256'] in lookup:
            j = lookup[r['sha256']]
            for k in old: arrays[k][i] = old[k][j]
            arrays['raw'][i] = oldraw[j]
        else: remaining.append(i)
    back = gf.BackboneRunner(run); start_time = time.perf_counter()
    for start in range(0, len(remaining), 16):
        ix = remaining[start:start + 16]
        pp = [gf.preprocess_image_views(gf.load_image(ROOT / 'الملفات_المهمة' / d[i]['relative_path'])) for i in ix]
        xc = np.stack([p[0] for p in pp]); xf = np.stack([p[1] for p in pp]); raw = np.stack([p[2] for p in pp])
        a, b, c, e = [back.run_view(v) for v in (xc, xc[:, :, :, ::-1].copy(), xf, xf[:, :, :, ::-1].copy())]
        dom, cr, fu, x = gf.build_descriptors_from_backbone_outputs(a[0], a[1], a[2], a[3], a[4], b[2], b[3], b[4], c[2], c[3], c[4], e[2], e[3], e[4], raw, stats['vs_mean'], stats['vs_std'])
        for k, value in [('domain', dom), ('crop', cr), ('full', fu), ('x53', x), ('raw', raw)]: arrays[k][ix] = value
        if start % 320 == 0: print('EXTRACT', start + len(ix), '/', len(remaining), 'seconds', round(time.perf_counter() - start_time, 1), flush=True)
    for a in arrays.values(): a.flush()
    js(run / 'cache/feature_identity.json', {'reused_exact_hash_images': len(d) - len(remaining), 'extracted': len(remaining),
        'seconds': time.perf_counter() - start_time, 'files': {k + '.npy': digest(run / 'cache' / (k + '.npy')) for k in dims}})
    validate(run)


def gram(run):
    validate(run); d = rows(run / 'images.csv'); n = len(d)
    if (run / 'cache/gram_identity.json').exists(): raise ValueError('مصفوفات التشابه موجودة')
    identities = js(run / 'cache/feature_identity.json')['files']
    for k in ('crop', 'full', 'x53'):
        p = run / 'cache' / (k + '.npy')
        if digest(p) != identities[p.name]: raise ValueError('اختلاف الميزات')
        x = np.array(np.load(p, mmap_mode='r'), dtype=np.float32)
        if k != 'full': x = x[:, :-154].copy()
        x /= np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
        g = np.lib.format.open_memmap(run / 'cache' / ('gram_' + k + '.npy'), mode='w+', dtype='float32', shape=(n, n))
        tt = time.perf_counter()
        for start in range(0, n, 256):
            end = min(n, start + 256); g[start:end] = x[start:end] @ x.T
            if start % 2048 == 0: print('GRAM', k, end, '/', n, 'seconds', round(time.perf_counter() - tt, 1), flush=True)
        g.flush(); del x, g
    js(run / 'cache/gram_identity.json', {'files': {k: digest(run / 'cache' / ('gram_' + k + '.npy')) for k in ('crop', 'full', 'x53')}})


def kernel(g, raw, ai, bi, mean, std):
    def vn(ii):
        x = np.clip((raw[ii].astype(np.float64) - mean) / std, -5, 5)
        return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
    tail = vn(ai) @ vn(bi).T
    sw = .1 / 1.95
    crop = (1 - sw) * np.asarray(g['crop'][np.ix_(ai, bi)], dtype=np.float64) + sw * tail
    x53 = np.clip((1 - sw) * np.asarray(g['x53'][np.ix_(ai, bi)], dtype=np.float64) + sw * tail, -1, 1)
    full = np.asarray(g['full'][np.ix_(ai, bi)], dtype=np.float64)
    return .55 * (.72 * crop + .28 * full) + .25 * x53 ** 2 + .20 * np.exp(-1.5 * np.maximum(2 - 2 * x53, 0))


def sample_weights(d, ix):
    counts = collections.Counter(d[i]['listing_uid'] for i in ix)
    return np.array([1. / counts[d[i]['listing_uid']] for i in ix])


def fit(K, target, weights, C):
    model = SVR(C=C, epsilon=.05, kernel='precomputed', cache_size=256, tol=.001)
    model.fit(K, np.log(target), sample_weight=weights)
    return model, np.log([target.min(), target.max()])


def bootstrap(y, p, reference, groups, repetitions=2000):
    unique, inv = np.unique(groups, return_inverse=True)
    sizes = np.bincount(inv); e = np.bincount(inv, weights=np.abs(y - p))
    delta = np.bincount(inv, weights=np.abs(y - p) - np.abs(y - reference))
    rng = np.random.default_rng(SEED); maes = []; diffs = []
    for _ in range(repetitions):
        ix = rng.integers(0, len(unique), len(unique)); denom = sizes[ix].sum()
        maes.append(e[ix].sum() / denom); diffs.append(delta[ix].sum() / denom)
    return {'MAE_95CI': np.quantile(maes, [.025, .975]).tolist(),
            'new_minus_reference_MAE_95CI': np.quantile(diffs, [.025, .975]).tolist(),
            'sampling_unit': 'group_id', 'replicates': repetitions, 'n_groups': len(unique),
            'conditional_on_selected_hyperparameters': True}


def gate(y, new, old, threshold):
    a = met(y, new); b = met(y, old); expensive = y >= threshold
    tests = {'MAE_improves_at_least_5_percent': a['MAE_SAR'] <= .95 * b['MAE_SAR'],
             'MAPE_not_worse_by_over_0_5pp': a['MAPE_percent'] <= b['MAPE_percent'] + .5,
             'hit20_not_worse_by_over_1pp': a['within_20_percent'] >= b['within_20_percent'] - 1,
             'expensive_MAE_not_worse_by_over_5_percent': np.mean(np.abs(y[expensive] - new[expensive])) <= 1.05 * np.mean(np.abs(y[expensive] - old[expensive]))}
    return {k: bool(v) for k, v in tests.items()}


def train(run):
    validate(run)
    if (run / 'CV_LOCK.json').exists(): raise ValueError('انتهى اختيار المرشحين')
    d = rows(run / 'images.csv'); protocol = js(run / 'protocol.json')
    raw = np.load(run / 'cache/raw.npy', mmap_mode='r')
    g = {}
    for key, sha in js(run / 'cache/gram_identity.json')['files'].items():
        p = run / 'cache' / ('gram_' + key + '.npy')
        if digest(p) != sha: raise ValueError('اختلاف مصفوفة التشابه')
        g[key] = np.load(p, mmap_mode='r')
    covers = np.array([i for i, r in enumerate(d) if r['split'] == 'train' and r['is_cover'] == 'True'])
    alltrain = np.array([i for i, r in enumerate(d) if r['split'] == 'train'])
    y = np.array([float(r['price_sar']) for r in d]); yy = y[covers]
    groups = np.array([d[i]['cv_group'] for i in covers]); grid = protocol['grid_C']
    oof = {(arm, C): np.zeros(len(covers)) for arm in ('single', 'multi') for C in grid}
    log = []; foldrows = []
    for fold, (fit_ix, hold_ix) in enumerate(GroupKFold(3).split(covers, yy, groups)):
        fitcovers = covers[fit_ix]; hold = covers[hold_ix]; fitgroups = {d[i]['cv_group'] for i in fitcovers}
        mean = raw[fitcovers].astype(np.float64).mean(0, keepdims=True)
        std = np.maximum(raw[fitcovers].astype(np.float64).std(0, keepdims=True), 1e-6)
        for i in hold: foldrows.append({'image_uid': d[i]['image_uid'], 'group_id': d[i]['group_id'], 'cv_group': d[i]['cv_group'], 'fold': fold})
        for arm in ('single', 'multi'):
            ai = fitcovers if arm == 'single' else np.array([i for i in alltrain if d[i]['cv_group'] in fitgroups])
            if {d[i]['cv_group'] for i in ai} & {d[i]['cv_group'] for i in hold}: raise ValueError('تسرب CV')
            w = sample_weights(d, ai); K = kernel(g, raw, ai, ai, mean, std); KH = kernel(g, raw, hold, ai, mean, std)
            for C in grid:
                tt = time.perf_counter(); model, bounds = fit(K, y[ai], w, C)
                pred = np.exp(np.clip(model.predict(KH), *bounds)); oof[arm, C][hold_ix] = pred
                log.append({'stage': 'train_GroupKFold', 'arm': arm, 'fold': fold, 'C': C, 'fit_images': len(ai),
                    'fit_listings': len({d[i]['listing_uid'] for i in ai}), 'weight_sum': float(w.sum()),
                    'seconds': time.perf_counter() - tt, **met(y[hold], pred)})
                csvwrite(run / 'experiments.csv', log)
                print('CV', arm, 'fold', fold + 1, 'C', C, 'MAE', round(log[-1]['MAE_SAR'], 2), flush=True)
            del K, KH
    csvwrite(run / 'fold_assignments.csv', foldrows)
    scores = [{'arm': arm, 'C': C, **met(yy, oof[arm, C])} for arm in ('single', 'multi') for C in grid]
    csvwrite(run / 'cv_summary.csv', scores)
    choices = {arm: min((r for r in scores if r['arm'] == arm), key=lambda r: (r['MAE_SAR'], r['C'])) for arm in ('single', 'multi')}
    paired = bootstrap(yy, oof['multi', choices['multi']['C']], oof['single', choices['single']['C']], [d[i]['group_id'] for i in covers])
    cvgate = gate(yy, oof['multi', choices['multi']['C']], oof['single', choices['single']['C']], np.quantile(yy, 2 / 3))
    csvwrite(run / 'oof_predictions.csv', [{'image_uid': d[i]['image_uid'], 'listing_uid': d[i]['listing_uid'], 'group_id': d[i]['group_id'],
        'actual_sar': float(y[i]), 'single_sar': float(oof['single', choices['single']['C']][j]),
        'multi_sar': float(oof['multi', choices['multi']['C']][j])} for j, i in enumerate(covers)])
    js(run / 'CV_LOCK.json', {'selected': choices, 'paired_bootstrap': paired, 'practical_gate': cvgate,
        'passed_practical_gate': all(cvgate.values()), 'val_not_used_for_selection': True,
        'test_inference_run': False, 'test_price_targets_used': False,
        'test_thumbnails_used_only_for_duplicate_audit': True,
        'protocol_sha256': digest(run / 'protocol.json'), 'oof_sha256': digest(run / 'oof_predictions.csv')})
    # Exactly two final fits; no val-based hyperparameter revision.
    val = np.array([i for i, r in enumerate(d) if r['split'] == 'val'])
    mean = raw[covers].astype(np.float64).mean(0, keepdims=True)
    std = np.maximum(raw[covers].astype(np.float64).std(0, keepdims=True), 1e-6)
    for arm in ('single', 'multi'):
        ai = covers if arm == 'single' else alltrain
        K = kernel(g, raw, ai, ai, mean, std); KV = kernel(g, raw, val, ai, mean, std)
        model, bounds = fit(K, y[ai], sample_weights(d, ai), choices[arm]['C'])
        pred = np.exp(np.clip(model.predict(KV), *bounds))
        train_pred = np.exp(np.clip(model.predict(K), *bounds))
        train_cover_mask = np.array([d[i]['is_cover'] == 'True' for i in ai])
        js(run / (arm + '_train_head_metrics.json'), {
            'in_sample_fixed_covers': met(y[ai][train_cover_mask], train_pred[train_cover_mask]),
            'out_of_fold_fixed_covers': choices[arm],
            'scope': 'head only, no Gemini blend; in-sample error is not generalization evidence'})
        csvwrite(run / (arm + '_val_head_predictions.csv'), [{'image_uid': d[i]['image_uid'], 'listing_uid': d[i]['listing_uid'],
            'group_id': d[i]['group_id'], 'is_cover': d[i]['is_cover'], 'actual_sar': float(y[i]), 'predicted_sar': float(p)} for i, p in zip(val, pred)])
        if arm == 'multi': export(run, d, ai, model, bounds, mean, std)
        del K, KV
    js(run / 'training_completed.json', {'two_final_arms_only': True, 'old_test_not_run': True})
    validate(run)


def export(run, d, ai, model, bounds, mean, std):
    # Export primal linear terms and support vectors; no listing table at runtime.
    gf = gf_module(run); raw = np.load(run / 'cache/raw.npy', mmap_mode='r')
    coef = np.zeros(len(ai)); coef[model.support_] = model.dual_coef_[0]
    def adjusted(key):
        x = np.array(np.load(run / 'cache' / (key + '.npy'), mmap_mode='r')[ai], dtype=np.float64)
        if key != 'full':
            x[:, :-154] *= np.sqrt(1 - .1 / 1.95) / np.maximum(np.linalg.norm(x[:, :-154], axis=1, keepdims=True), 1e-12)
            v = np.clip((raw[ai].astype(np.float64) - mean) / std, -5, 5)
            x[:, -154:] = gf.l2_norm(v) * np.sqrt(.1 / 1.95)
            x = gf.l2_norm(x)
        return x
    crop = adjusted('crop'); full = adjusted('full'); x53 = adjusted('x53')
    np.savez_compressed(run / 'head.npz', w_crop=(.55 * .72 * crop.T @ coef), w_full=(.55 * .28 * full.T @ coef),
        support53=x53[model.support_].astype(np.float32), poly_coef=.25 * model.dual_coef_[0],
        rbf_coef=.20 * model.dual_coef_[0], intercept=model.intercept_, vs_mean=mean, vs_std=std)
    shutil.copytree(BASE / 'gemini', run / 'gemini', ignore=shutil.ignore_patterns('val_predictions.csv'))
    shutil.copy2(ROOT / 'scripts/image_price_multiview_runtime.py', run / 'runtime.py')
    js(run / 'policy.json', {'status': 'experimental_candidate', 'bounds': bounds.tolist(), 'gemini_weight': .25,
        'pixels_only': True, 'single_image_inference': True, 'listing_database_required': False,
        'domain_heuristic_unchanged': True, 'external_validation': 'لم يختبر بعد'})
    print('EXPORTED multi support vectors', len(model.support_), flush=True)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('command', choices=['prepare', 'extract', 'gram', 'train'])
    ap.add_argument('--run', type=Path); a = ap.parse_args()
    if a.command == 'prepare': return prepare()
    run = a.run or ROOT / js(ROOT / 'models/image_price_multiview_work.json')['run_dir']
    {'extract': extract, 'gram': gram, 'train': train}[a.command](run)


if __name__ == '__main__': main()
