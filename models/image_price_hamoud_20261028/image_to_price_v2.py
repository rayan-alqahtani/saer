"""
Saer - Image-only price estimation  (image -> price)
===========================================================================
Input  : car photos ONLY (no make / model / year / mileage / other models' outputs)
Output : estimated price (SAR) + calibrated 80% / 90% range

Highlights
  1. Stronger / more backbones: DINOv2 small/base/large + CLIP ViT-B/16 (+ concat of all)
  2. Two training levels: per-listing (mean of photos) and per-IMAGE (then average per listing)
  3. Horizontal-flip augmentation in training + flip TTA at prediction
  4. Model selection on VAL only (single best vs ensemble of top-3 / top-5)
  5. Final refit on train+val, TEST evaluated once
  6. Conformal-style prediction range from 5-fold out-of-fold residuals (coverage checked on TEST)

CLI (stages are cached, so they are resumable):
  python image_to_price_v2.py embed    --zip images.zip --csv data.csv
  python image_to_price_v2.py search   --zip images.zip --csv data.csv
  python image_to_price_v2.py finalize --zip images.zip --csv data.csv
"""
import argparse
import json
import time
import zipfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

SEED = 42
OUT = Path("price_v2_outputs")
CACHE = OUT / "emb_cache"

BACKBONES = {
    "dinov2_s": ("hub", "dinov2_vits14"),
    "dinov2_b": ("hub", "dinov2_vitb14"),
    "dinov2_l": ("hub", "dinov2_vitl14"),
    "clip_b16": ("clip", "openai/clip-vit-base-patch16"),
}
FEATSETS = {**{k: [k] for k in BACKBONES}, "concat_all": list(BACKBONES)}
GRID = {
    "listing": ["ridge10", "ridge100", "ridge1000", "pca64_hgb", "svr"],
    "image": ["ridge100", "ridge500", "ridge2500", "pca64_hgb"],
}


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------
def extract_zip(zip_path, out_dir):
    out_dir = Path(out_dir)
    if out_dir.exists() and any(out_dir.rglob("*.webp")):
        return out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(out_dir)
    return out_dir


def build_image_table(csv_path, img_root):
    """One row per image. Matching by folder names (source_dataset/source_listing_id)."""
    df = pd.read_csv(csv_path)
    df = df[df["usable_price"]].copy()
    df["key"] = df["source_dataset"] + "|" + df["source_listing_id"].astype(str)
    found = {}
    for p in Path(img_root).rglob("*.webp"):
        found.setdefault(p.parent.parent.name + "|" + p.parent.name, []).append(p)
    rows = []
    for _, r in df.iterrows():
        for p in sorted(found.get(r["key"], [])):
            rows.append(dict(path=str(p), listing_uid=r["listing_uid"],
                             price=float(r["sar_price"]), split=r["split"]))
    imgs = pd.DataFrame(rows)
    print(f"[data] listings={imgs.listing_uid.nunique()} images={len(imgs)}")
    return imgs


def prepare(csv, zip_path, extract_to):
    root = extract_zip(zip_path, extract_to)
    imgs = build_image_table(csv, root).reset_index(drop=True)
    li, uniq = pd.factorize(imgs["listing_uid"])
    meta = imgs.drop_duplicates("listing_uid").set_index("listing_uid").loc[uniq]
    return imgs, li, uniq, meta["price"].values.astype(float), meta["split"].values


# ----------------------------------------------------------------------------
# Embeddings (original + horizontally flipped), cached on disk
# ----------------------------------------------------------------------------
class _PathDS:
    def __init__(self, paths, tf):
        self.paths, self.tf = list(paths), tf

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        from PIL import Image
        return self.tf(Image.open(self.paths[i]).convert("RGB"))


def _run_encoder(name, paths, batch=64, workers=2):
    import torch
    import torchvision.transforms as T
    kind, ident = BACKBONES[name]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if kind == "hub":
        enc = torch.hub.load("facebookresearch/dinov2", ident)
        mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        fwd = lambda x: enc(x)
    else:
        from transformers import CLIPModel
        enc = CLIPModel.from_pretrained(ident)
        mean, std = (0.4815, 0.4578, 0.4082), (0.2686, 0.2613, 0.2758)
        fwd = lambda x: enc.vision_model(pixel_values=x).pooler_output
    enc = enc.eval().to(dev)
    tf = T.Compose([T.Resize((224, 224), interpolation=T.InterpolationMode.BICUBIC),
                    T.ToTensor(), T.Normalize(mean, std)])
    dl = torch.utils.data.DataLoader(_PathDS(paths, tf), batch_size=batch,
                                     num_workers=workers)
    E, Ef = [], []
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16,
                                         enabled=(dev == "cuda")):
        for i, x in enumerate(dl):
            x = x.to(dev)
            E.append(fwd(x).float().cpu().numpy())
            Ef.append(fwd(torch.flip(x, dims=[3])).float().cpu().numpy())
            if i % 20 == 0:
                print(f"  [{name}] batch {i}/{len(dl)}")
    del enc
    if dev == "cuda":
        torch.cuda.empty_cache()
    return np.concatenate(E), np.concatenate(Ef)


def get_embeddings(name, paths):
    CACHE.mkdir(parents=True, exist_ok=True)
    f = CACHE / f"{name}.npz"
    pa = np.array(paths)
    if f.exists():
        d = np.load(f, allow_pickle=True)
        if len(d["paths"]) == len(pa) and (d["paths"] == pa).all():
            print(f"[embed] {name}: loaded from cache")
            return d["E"], d["Ef"]
    t = time.time()
    E, Ef = _run_encoder(name, paths)
    np.savez_compressed(f, E=E, Ef=Ef, paths=pa)
    print(f"[embed] {name}: {E.shape} done in {time.time() - t:.0f}s")
    return E, Ef


def l2norm(x):
    return (x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)).astype(np.float32)


def load_all(imgs):
    emb = {}
    for n in BACKBONES:
        E, Ef = get_embeddings(n, imgs["path"].tolist())
        emb[n] = (l2norm(E), l2norm(Ef))
    return emb


# ----------------------------------------------------------------------------
# Modeling
# ----------------------------------------------------------------------------
def metrics(y_true, y_pred):
    return dict(
        MAE=float(mean_absolute_error(y_true, y_pred)),
        MedAPE=float(np.median(np.abs(y_pred - y_true) / y_true) * 100),
        R2=float(r2_score(y_true, y_pred)),
        R2_log=float(r2_score(np.log(y_true), np.log(y_pred))),
    )


def pool(li, nl, A):
    cnt = np.bincount(li, minlength=nl)
    M = csr_matrix((1.0 / cnt[li], (li, np.arange(len(li)))), shape=(nl, len(li)))
    return np.asarray(M @ A, dtype=np.float32)


def build_F(featset, emb, li, nl, ylog):
    bbs = FEATSETS[featset]
    E = np.hstack([emb[b][0] for b in bbs])
    Ef = np.hstack([emb[b][1] for b in bbs])
    return dict(E=E, Ef=Ef, li=li, Xl=pool(li, nl, E), Xlf=pool(li, nl, Ef), ylog=ylog)


def make_model(kind, d):
    if kind.startswith("ridge"):
        return make_pipeline(StandardScaler(), Ridge(alpha=float(kind[5:])))
    if kind == "pca64_hgb":
        return make_pipeline(
            StandardScaler(), PCA(min(64, d), random_state=SEED),
            HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_depth=4,
                                          l2_regularization=1.0, random_state=SEED))
    if kind == "svr":
        return make_pipeline(StandardScaler(), PCA(min(128, d), random_state=SEED),
                             SVR(C=3.0, epsilon=0.1, gamma="scale"))
    raise ValueError(kind)


def fit_component(F, level, kind, fit_li):
    """Train on `fit_li` listings (orig + flipped embeddings = 2x augmentation)."""
    ylog = F["ylog"]
    if level == "listing":
        X = np.vstack([F["Xl"][fit_li], F["Xlf"][fit_li]])
        y = np.concatenate([ylog[fit_li]] * 2)
    else:
        m = np.isin(F["li"], fit_li)
        X = np.vstack([F["E"][m], F["Ef"][m]])
        y = np.concatenate([ylog[F["li"][m]]] * 2)
    mdl = make_model(kind, X.shape[1])
    mdl.fit(X, y)
    return mdl


def predict_component(mdl, F, level, pred_li):
    """Log-price prediction per listing, with flip TTA."""
    if level == "listing":
        return 0.5 * (mdl.predict(F["Xl"][pred_li]) + mdl.predict(F["Xlf"][pred_li]))
    m = np.isin(F["li"], pred_li)
    p = 0.5 * (mdl.predict(F["E"][m]) + mdl.predict(F["Ef"][m]))
    s = pd.Series(p).groupby(F["li"][m]).mean()
    return s.reindex(pred_li).values


# ----------------------------------------------------------------------------
# Stage: embed
# ----------------------------------------------------------------------------
def cmd_embed(a):
    OUT.mkdir(exist_ok=True)
    imgs, *_ = prepare(a.csv, a.zip, a.extract_to)
    load_all(imgs)
    print("[embed] all backbones ready")


# ----------------------------------------------------------------------------
# Stage: search (grid on TRAIN -> evaluate on VAL only)
# ----------------------------------------------------------------------------
def cmd_search(a):
    OUT.mkdir(exist_ok=True)
    imgs, li, uniq, price, split = prepare(a.csv, a.zip, a.extract_to)
    nl, ylog = len(uniq), np.log(price)
    emb = load_all(imgs)
    tr, va = np.where(split == "train")[0], np.where(split == "val")[0]

    rows, vp = [], {}
    for fs in FEATSETS:
        F = build_F(fs, emb, li, nl, ylog)
        for level, kinds in GRID.items():
            for kind in kinds:
                t = time.time()
                mdl = fit_component(F, level, kind, tr)
                p = predict_component(mdl, F, level, va)
                met = metrics(price[va], np.exp(p))
                rows.append(dict(feat=fs, level=level, kind=kind, **met,
                                 seconds=round(time.time() - t, 1)))
                vp[f"{fs}|{level}|{kind}"] = p
                print(f"[search] {fs:>10} {level:>7} {kind:>10}  "
                      f"VAL MAE={met['MAE']:>8,.0f}  MedAPE={met['MedAPE']:5.1f}%  "
                      f"R2={met['R2']:.3f}  ({rows[-1]['seconds']}s)")
        del F
    tbl = pd.DataFrame(rows).sort_values("MAE").reset_index(drop=True)
    tbl.to_csv(OUT / "val_table.csv", index=False)
    np.savez_compressed(OUT / "val_preds.npz", **{k.replace("|", "__"): v for k, v in vp.items()})
    print("\n===== TOP 10 on VAL =====")
    print(tbl.head(10).round(3).to_string(index=False))


# ----------------------------------------------------------------------------
# Stage: finalize (ensemble pick on VAL -> refit train+val -> TEST once -> conformal)
# ----------------------------------------------------------------------------
def pick_selection(tbl, vp, y_val):
    best = tbl.sort_values("MAE").groupby(["feat", "level"]).head(1).sort_values("MAE")
    keys = [f"{r.feat}|{r.level}|{r.kind}" for r in best.itertuples()]
    options = {"single_best": keys[:1], "ensemble_top3": keys[:3], "ensemble_top5": keys[:5]}
    scored = {}
    for name, ks in options.items():
        p = np.mean([vp[k] for k in ks], axis=0)
        scored[name] = dict(keys=ks, **metrics(y_val, np.exp(p)))
    choice = min(scored, key=lambda n: scored[n]["MAE"])   # chosen on VAL only
    return choice, scored


def ens_logpred(comps_models, Fs, pred_li):
    return np.mean([predict_component(m, Fs[fs], lvl, pred_li)
                    for (fs, lvl, _), m in comps_models], axis=0)


def cmd_finalize(a):
    imgs, li, uniq, price, split = prepare(a.csv, a.zip, a.extract_to)
    nl, ylog = len(uniq), np.log(price)
    emb = load_all(imgs)
    tr, va, te = (np.where(split == s)[0] for s in ("train", "val", "test"))
    trval = np.concatenate([tr, va])

    tbl = pd.read_csv(OUT / "val_table.csv")
    vpz = np.load(OUT / "val_preds.npz")
    vp = {k.replace("__", "|"): vpz[k] for k in vpz.files}
    choice, scored = pick_selection(tbl, vp, price[va])
    comps = [tuple(k.split("|")) for k in scored[choice]["keys"]]
    print(f"[final] selected on VAL: {choice}")
    for k in scored[choice]["keys"]:
        print("   -", k)
    print("   VAL:", {m: round(v, 3) for m, v in scored[choice].items() if m != "keys"})

    Fs = {fs: build_F(fs, emb, li, nl, ylog) for fs in {c[0] for c in comps}}

    # (a) train-only fit -> TEST
    m_tr = [(c, fit_component(Fs[c[0]], c[1], c[2], tr)) for c in comps]
    test_train_only = metrics(price[te], np.exp(ens_logpred(m_tr, Fs, te)))

    # (b) final refit on train+val -> TEST (this one is deployed)
    m_fin = [(c, fit_component(Fs[c[0]], c[1], c[2], trval)) for c in comps]
    lp_te = ens_logpred(m_fin, Fs, te)
    test_final = metrics(price[te], np.exp(lp_te))

    # (c) conformal-style range: 5-fold out-of-fold residuals over train+val
    oof = np.zeros(len(trval))
    for k, (a_idx, b_idx) in enumerate(KFold(5, shuffle=True, random_state=SEED).split(trval)):
        fit_li, hold_li = trval[a_idx], trval[b_idx]
        mm = [(c, fit_component(Fs[c[0]], c[1], c[2], fit_li)) for c in comps]
        oof[b_idx] = ens_logpred(mm, Fs, hold_li)
        print(f"[final] CV fold {k + 1}/5 done")
    r = ylog[trval] - oof
    q = dict(lo80=float(np.quantile(r, .10)), hi80=float(np.quantile(r, .90)),
             lo90=float(np.quantile(r, .05)), hi90=float(np.quantile(r, .95)))

    pred = np.exp(lp_te)
    cov = {f"coverage{c}": float(np.mean((price[te] >= pred * np.exp(q[f"lo{c}"])) &
                                         (price[te] <= pred * np.exp(q[f"hi{c}"]))))
           for c in (80, 90)}
    pd.DataFrame(dict(
        listing_uid=uniq[te], true=price[te], pred=pred,
        lo80=pred * np.exp(q["lo80"]), hi80=pred * np.exp(q["hi80"]),
        lo90=pred * np.exp(q["lo90"]), hi90=pred * np.exp(q["hi90"]),
    )).to_csv(OUT / "test_predictions.csv", index=False)

    bundle = dict(components=[dict(featset=c[0], backbones=FEATSETS[c[0]], level=c[1],
                                   kind=c[2], model=m) for c, m in m_fin], q=q)
    joblib.dump(bundle, OUT / "bundle.joblib")

    dummy = np.median(price[tr])
    res = dict(selection=choice, components=scored[choice]["keys"],
               val_all_options={k: {m: v for m, v in d.items() if m != "keys"} | {"keys": d["keys"]}
                                for k, d in scored.items()},
               dummy_test=metrics(price[te], np.full(len(te), dummy)),
               test_train_only=test_train_only, test_final=test_final,
               conformal_q=q, **cov)
    json.dump(res, open(OUT / "results_v2.json", "w"), indent=2, ensure_ascii=False)

    print("\n================ FINAL (TEST, 345 listings) ================")
    print(f"{'Dummy (median)':<28}", {k: round(v, 3) for k, v in res["dummy_test"].items()})
    print(f"{'v2 train-only fit':<28}", {k: round(v, 3) for k, v in test_train_only.items()})
    print(f"{'v2 FINAL (train+val)':<28}", {k: round(v, 3) for k, v in test_final.items()})
    print(f"80% range coverage on TEST: {cov['coverage80']:.1%}   90%: {cov['coverage90']:.1%}")


# ----------------------------------------------------------------------------
# Predict for an external listing (any photos, not from the dataset)
# ----------------------------------------------------------------------------
def comp_logpred(comp, emb_ext):
    E = np.hstack([emb_ext[b][0] for b in comp["backbones"]])
    Ef = np.hstack([emb_ext[b][1] for b in comp["backbones"]])
    mdl = comp["model"]
    if comp["level"] == "listing":
        return 0.5 * (mdl.predict(E.mean(0, keepdims=True)) +
                      mdl.predict(Ef.mean(0, keepdims=True)))[0]
    return float(np.mean(0.5 * (mdl.predict(E) + mdl.predict(Ef))))


def predict_external(paths, bundle_path=OUT / "bundle.joblib"):
    b = joblib.load(bundle_path)
    need = sorted({bb for c in b["components"] for bb in c["backbones"]})
    emb = {}
    for n in need:
        E, Ef = _run_encoder(n, list(paths), workers=0)
        emb[n] = (l2norm(E), l2norm(Ef))
    lp = float(np.mean([comp_logpred(c, emb) for c in b["components"]]))
    q = b["q"]
    return dict(price=float(np.exp(lp)),
                lo80=float(np.exp(lp + q["lo80"])), hi80=float(np.exp(lp + q["hi80"])),
                lo90=float(np.exp(lp + q["lo90"])), hi90=float(np.exp(lp + q["hi90"])))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    for name, fn in [("embed", cmd_embed), ("search", cmd_search), ("finalize", cmd_finalize)]:
        s = sp.add_parser(name)
        s.add_argument("--zip", required=True)
        s.add_argument("--csv", required=True)
        s.add_argument("--extract_to", default="/content/images_extracted")
        s.set_defaults(fn=fn)
    a = ap.parse_args()
    a.fn(a)
