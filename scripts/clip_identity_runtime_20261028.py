"""Standalone tester of only the CLIP heads that passed the development gate."""
from __future__ import annotations
import json
import numpy as np
import joblib
import torch
from PIL import Image
from scipy.special import softmax
import clip_identity_round_20261027 as clip
from model_image_tester_20261026 import CandidateRuntime
from image_input_policy_20261026 import decode_rgb
import visual_fusion_predict as base


class ClipIdentityRuntime(CandidateRuntime):
    def __init__(self):
        super().__init__()
        self.clip_decisions=clip.r.v.load_json(clip.OUT/'results.json')
        self.clip_heads={t:joblib.load(clip.OUT/t/'head.joblib') for t,z in self.clip_decisions.items() if z['accepted']}
        if not self.clip_heads:
            raise ValueError('لم يجتز أي رأس CLIP قاعدة الاستبدال؛ استخدم نسخة26 أكتوبر')
        release=clip.r.v.load_json(clip.OUT/'runtime_release.json')
        for name,digest in release['files'].items():
            if clip.r.digest(clip.ROOT/name)!=digest:raise ValueError('تغير ملف تشغيل المرشح: '+name)
        self.clip_model,self.clip_processor=clip.load_clip()
        self.clip_text={t:np.load(clip.OUT/f'{t}_text.npy') for t in self.clip_heads}
        self.clip_scale=clip.r.v.load_json(clip.OUT/'extraction_complete.json')['text_logit_scale']
        self.root=clip.OUT

    def extract(self,path):
        f=super().extract(path)
        with Image.open(path) as source:im=decode_rgb(source)
        crop=im.crop(f['box']) if f['detected'] else im
        with torch.inference_mode():
            x=self.clip_processor(images=[im,crop],return_tensors='pt')
            f['_clip']=clip.norm(self.clip_model.get_image_features(**x).numpy())
        return f

    def clip_probabilities(self,task,f):
        art=self.clip_heads[task]
        original=self.candidate_features('make',f) if task=='make' else base.features(f,clip.r.v.load_json(clip.r.v.RUN/task/'selection_lock.json')['kind'],task)
        x=np.c_[original,f['_clip'][0][None],f['_clip'][1][None]]
        p=art['model'].predict_proba(art['scaler'].transform(x))
        semantic=clip.norm(f['_clip'].sum(axis=0)[None])@self.clip_text[task].T*self.clip_scale
        ps=softmax(semantic,axis=1)
        strength=np.clip(1.+(ps*np.log(np.maximum(ps,1e-12))).sum(1)/np.log(ps.shape[1]),0,1)
        p=softmax(np.log(np.maximum(p,1e-10))+art['text_weight']*strength[:,None]*(semantic-semantic.mean(1,keepdims=True)),axis=1)
        return p,art['classes']

    def predict(self,task,path,k=5,features=None):
        if task!='recognize':return super().predict(task,path,k,features)
        f=features if features is not None else self.extract(path)
        old=super().predict(task,path,k,f)
        pm,mc=self.clip_probabilities('make',f) if 'make' in self.clip_heads else self.probabilities('make',path,f)
        pc,cc=self.clip_probabilities('model',f) if 'model' in self.clip_heads else self.probabilities('model',path,f)
        mt=self._top(pm[0],mc);ct=self._top(pc[0],cc);make,_,model=ct[0]['الفئة'].partition('|')
        return {'الحالة':'مرشح تطويري CLIP + DINO',
            'الماركة_المستقلة':{'النتيجة':mt[0]['الفئة'],'أفضل_الاحتمالات':mt},
            'الموديل_المستقل':{'النتيجة':ct[0]['الفئة'],'أفضل_الاحتمالات':ct},
            'الاقتراح_الرئيسي':{'النتيجة':ct[0]['الفئة'],'الماركة':make,'الموديل':model,
                'درجة_غير_معايرة':ct[0]['درجة_غير_معايرة'],'أفضل_الاحتمالات':ct},
            'اتساق_الماركة_والموديل':make==mt[0]['الفئة'],
            'الموديل_قبل_الإضافة':old['الاقتراح_الرئيسي']['النتيجة'],
            'الماركة_قبل_الإضافة':old['الماركة_المستقلة']['النتيجة'],
            'المسار':'رؤوس CLIP المقبولة داخليًا فقط: '+','.join(self.clip_heads),
            'نطاق_الموديل':'الفئات المدعومة فقط؛ لا ضمان خارجي أو معايرة خارجية للدرجات',
            'يلزم_تأكيد_المستخدم_قبل_التسعير':True}


def main():
    import tkinter as tk
    from model_image_tester import ModelImageTester
    root=tk.Tk();root.title('سعّر — تجربة CLIP + DINO')
    ModelImageTester(root,engine_factory=ClipIdentityRuntime,
        subtitle='مرشح CLIP + DINO — النتائج تجريبية؛ أكّد السيارة قبل تقدير السعر')
    root.mainloop()


if __name__=='__main__':main()
