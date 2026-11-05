"""Local frozen CLIP representation experiment; never open historical test."""
from __future__ import annotations
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[k]='2'
os.environ['TOKENIZERS_PARALLELISM']='false'
import argparse, json, sys, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/clip_identity_20261027_v1'
sys.path.insert(0,str(OUT/'deps'))
import numpy as np
from PIL import Image
from image_input_policy_20261026 import decode_rgb
import robust_visual_round_20261026 as r

HF=Path.home()/'.cache/huggingface/hub/models--openai--clip-vit-base-patch32/snapshots'
CONFIG=HF/'3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268'
WEIGHTS=HF/'c237dc49a33fc61debc9276459120b7eac67e7ef/model.safetensors'
TEMPLATES=['a photo of a {} car.','a photo of the front of a {} car.',
           'a photo of the side of a {} car.','a photo of the rear of a {} car.']


def load_clip():
    import torch
    from transformers import CLIPModel, CLIPConfig, CLIPProcessor
    from safetensors.torch import load_file
    config=CLIPConfig.from_json_file(str(CONFIG/'config.json'))
    config._attn_implementation='sdpa'
    model=CLIPModel(config)
    state=load_file(str(WEIGHTS))
    # Older official checkpoint stored deterministic position buffers. Newer
    # Transformers regenerates them rather than treating them as parameters.
    removed=[]
    for key in ('text_model.embeddings.position_ids','vision_model.embeddings.position_ids'):
        if key in state:
            value=state.pop(key)
            assert torch.equal(value,torch.arange(value.numel(),dtype=value.dtype).reshape(value.shape))
            removed.append(key)
    model.load_state_dict(state,strict=True)
    compatibility={'removed_deterministic_buffers':removed,
        'validated_values_are_arange':True,'all_learned_weights_loaded_strictly':True,'original_unchanged':True}
    if (OUT/'checkpoint_compatibility.json').exists():
        assert r.v.load_json(OUT/'checkpoint_compatibility.json')==compatibility
    else:r.save(OUT/'checkpoint_compatibility.json',compatibility)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device).eval().requires_grad_(False)
    processor=CLIPProcessor.from_pretrained(str(CONFIG),local_files_only=True)
    return model,processor


def norm(x):
    return x/np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-12)


def extract():
    import torch
    torch.set_num_threads(4)
    rows=r.v.compact_rows('make','train')+r.v.compact_rows('make','val')
    assert all(z['split'] in ('train','val') for z in rows)
    dest=OUT/'features'; dest.mkdir(exist_ok=True)
    if not (OUT/'protocol.json').exists():
        protect={str(p.relative_to(ROOT)):r.digest(p) for p in (ROOT/'models').glob('*current.json')}
        r.save(OUT/'protocol.json',{'backbone':'openai/clip-vit-base-patch32 frozen local safetensors',
          'weights_sha256':r.digest(WEIGHTS),'config_sha256':r.digest(CONFIG/'config.json'),
          'preprocessing':'local official CLIPProcessor; EXIF/alpha normalized; full and saved car crop',
          'text_templates':TEMPLATES,'head_configs':['clip_plus_current_C0.1','clip_plus_current_C1.0'],
          'selection':'three outer existing train group folds; three inner folds choose C; one final candidate per task on historical val after train gate',
          'train_gate':'macro-F1 gain >=.01; no accuracy decline >.005; paired group bootstrap lower>0',
          'val_gate':'macro-F1 gain >=.01; no accuracy decline >.005; paired group bootstrap lower>0 against full current candidate runtime',
          'primary':'single-image predictions across all views; equal total listing weight',
          'secondary':'one fixed image per listing, plus frozen zero-shot reference',
          'test_read':False,'external_images_used_for_selection':False,'protected':protect})
    if (dest/'index.csv').exists():
        prior=r.v.read_csv(dest/'index.csv')
        assert [z['image_uid'] for z in prior]==[z['image_uid'] for z in rows]
    else:r.table(dest/'index.csv',rows)
    n=len(rows); progress=r.v.load_json(dest/'progress.json') if (dest/'progress.json').exists() else {'done':0}
    full=np.lib.format.open_memmap(dest/'full.npy',mode='r+' if (dest/'full.npy').exists() else 'w+',dtype='float32',shape=(n,512))
    crop=np.lib.format.open_memmap(dest/'crop.npy',mode='r+' if (dest/'crop.npy').exists() else 'w+',dtype='float32',shape=(n,512))
    boxes={z['sha256']:z for z in r.v.read_csv(r.v.RUN/'features/boxes.csv')}
    model,processor=load_clip(); started=time.perf_counter()
    print('CLIP_LOADED','images',n,'resume',progress['done'],flush=True)
    with torch.inference_mode():
        for st in range(progress['done'],n,16):
            batch=rows[st:st+16]; ims=[]; cropped=[]
            for z in batch:
                with Image.open(ROOT/'الملفات_المهمة'/z['relative_path']) as src:im=decode_rgb(src)
                box=boxes[z['sha256']]
                ims.append(im)
                cropped.append(im.crop(json.loads(box['box'])) if box['found']=='True' else im)
            inp=processor(images=ims+cropped,return_tensors='pt')
            features=norm(model.get_image_features(**inp).numpy())
            full[st:st+len(batch)]=features[:len(batch)]
            crop[st:st+len(batch)]=features[len(batch):]
            if st%128==0 or st+len(batch)==n:
                full.flush();crop.flush()
                r.save(dest/'progress.json',{'done':st+len(batch),'total':n,'current_invocation_seconds':time.perf_counter()-started})
                print('CLIP_FEATURES',st+len(batch),'/',n,'seconds',round(time.perf_counter()-started,1),flush=True)
        classes={t:sorted({z[r.TARGET[t]] for z in r.v.compact_rows(t,'train')}) for t in ('make','model')}
        r.save(OUT/'classes.json',classes)
        for task,cs in classes.items():
            descriptions=[template.format(c.replace('|',' ')) for c in cs for template in TEMPLATES]
            tokens=processor(text=descriptions,return_tensors='pt',padding=True,truncation=True)
            text=norm(model.get_text_features(**tokens).numpy()).reshape(len(cs),len(TEMPLATES),512).mean(1)
            np.save(OUT/f'{task}_text.npy',norm(text))
        r.save(OUT/'extraction_complete.json',{'images':n,'weight_downloaded':False,'frozen':True,'eval_mode':True,'no_grad':True,
                'text_logit_scale':float(model.logit_scale.exp()),'weights_sha256':r.digest(WEIGHTS)})
    r.check_protected(r.v.load_json(OUT/'protocol.json')['protected'])


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['extract']);args=parser.parse_args()
    extract()
