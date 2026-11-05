"""Separate tester for internally qualified development candidates from Oct 26."""
import json
import numpy as np
import joblib
from PIL import Image
import tkinter as tk
import torch
import visual_fusion_predict as base
from model_image_tester import ModelImageTester,VisualFusionRuntime
import foreground_visual_round_20261026 as fg
import geometry_visual_round_20261026 as geo

class CandidateRuntime(VisualFusionRuntime):
    def __init__(self):
        super().__init__()
        self.root=fg.r.ROOT/'outputs/visual_rescue_20261026'
        manifest=fg.r.v.load_json(self.root/'runtime_code_manifest.json')
        for name,sha in manifest['files'].items():
            if fg.r.digest(fg.r.ROOT/name)!=sha:raise ValueError('تغير ملف تشغيل المرشح: '+name)
        self.selection=fg.r.v.load_json(self.root/'chosen_development_candidates.json')['tasks']
        self.artifacts={};self.foreground_models=None;self.geometry_models=None
        for task,entry in self.selection.items():
            path=fg.r.ROOT/entry['artifact']
            if fg.r.digest(path)!=entry['sha256']:raise ValueError('تغير وزن المرشح: '+task)
            self.artifacts[task]=joblib.load(path)
    def extract(self,path):
        f=super().extract(path)
        if any(z['family'] in ('foreground','foreground_geometry_fusion') for z in self.selection.values()):
            if self.foreground_models is None:self.foreground_models=fg.models()
            sha=fg.r.digest(path);row={'relative_path':str(path),'image_uid':path.name,'sha256':sha}
            b={sha:{'box':json.dumps(f['box']),'found':str(f['detected'])}}
            pp,cc,_,_,_,_=fg.batch_features([row],self.foreground_models,b)
            f['_foreground_patch']=pp[0];f['_foreground_paint']=cc[0]
        if any(z['family'] in ('preserve_geometry','foreground_geometry_fusion') for z in self.selection.values()):
            if self.geometry_models is None:self.geometry_models=geo.load_model()
            d,tf=self.geometry_models
            with Image.open(path) as src:im=geo.decode_rgb(src)
            crop=im.crop(f['box']) if f['detected'] else im
            with torch.inference_mode():f['_geometry']=d(tf(geo.letterbox(crop))[None]).numpy()[0]
        return f
    def candidate_features(self,task,f):
        art=self.artifacts[task];entry=self.selection[task]
        if entry['family']=='foreground':
            x=fg.single_image_features(f,f['_foreground_patch'],f['_foreground_paint'],task,art)
        elif entry['family']=='preserve_geometry':
            if art['config']['kind']=='letterbox':x=f['_geometry'][None]
            else:
                selected=fg.r.v.load_json(fg.r.v.RUN/task/'selection_lock.json')
                x=np.c_[base.features(f,selected['kind'],task),f['_geometry'][None]]
        elif entry['family']=='foreground_geometry_fusion':
            selected=fg.r.v.load_json(fg.r.v.RUN/task/'selection_lock.json')
            x=np.c_[base.features(f,selected['kind'],task),f['_foreground_patch'][None],f['_geometry'][None]]
        else:raise ValueError('عائلة مرشح غير مدعومة')
        return x
    def probabilities(self,task,path,f):
        old,cs=base.predict(task,path,f)
        if task not in self.selection:return old,cs
        art=self.artifacts[task];x=self.candidate_features(task,f)
        p=art['model'].predict_proba(art['scaler'].transform(x))
        assert list(cs)==list(art['classes'])
        if art.get('runtime_blend_policy') in ('half_reference','replace_direct_component_keep_half_reference'):
            direct,dc=base.direct(task,f);assert list(dc)==list(cs)
            p=old+.5*(p-direct);p=np.maximum(p,0);p/=p.sum(1,keepdims=True)
        return p,cs
    def predict(self,task,path,k=5,features=None):
        f=features if features is not None else self.extract(path)
        if task=='recognize':
            mp,mc=self.probabilities('make',path,f);cp,cc=self.probabilities('model',path,f)
            mt=self._top(mp[0],mc);ct=self._top(cp[0],cc);label=ct[0]['الفئة'];make,_,model=label.partition('|')
            return {'الحالة':'مرشح تطويري','الماركة_المستقلة':{'النتيجة':mt[0]['الفئة'],'أفضل_الاحتمالات':mt},
                'الموديل_المستقل':{'النتيجة':label,'أفضل_الاحتمالات':ct},
                'الاقتراح_الرئيسي':{'النتيجة':label,'الماركة':make,'الموديل':model,
                    'درجة_غير_معايرة':ct[0]['درجة_غير_معايرة'],'أفضل_الاحتمالات':ct},
                'اتساق_الماركة_والموديل':make==mt[0]['الفئة'],
                'المسار':'مرشحو 26 أكتوبر المقبولون داخليًا مع الاحتفاظ ببقية الرؤوس',
                'نطاق_الموديل':'لا ضمان للتعميم الخارجي؛ لم تُفرض ماركة رأس على الموديل.'}
        if task in ('color','body') and task in self.selection:
            p,cs=self.probabilities(task,path,f);top=self._top(p[0],cs)
            for z in top:z['الفئة']=self._arabic_label(task,z['الفئة'])
            return {'النتيجة':top[0]['الفئة'],'أفضل_النتائج':top,'الحالة':'مرشح تطويري 26 أكتوبر',
                'درجة_القرار':top[0]['درجة_غير_معايرة'],'تنبيه':'درجات غير معايرة؛ لم يثبت التعميم الخارجي.'}
        return super().predict(task,path,k,f)

def main():
    root=tk.Tk()
    selected=fg.r.v.load_json(fg.r.ROOT/'outputs/visual_rescue_20261026/chosen_development_candidates.json')['tasks']
    label={'make':'الماركة','model':'الموديل','color':'اللون','body':'الهيكل'}
    changes='، '.join(label[t] for t in selected)
    ModelImageTester(root,engine_factory=CandidateRuntime,
        subtitle=f'مرشح 26 أكتوبر — تحديث: {changes}. بقية الرؤوس محفوظة؛ التعميم الخارجي غير مثبت.')
    root.title('سعّر — تجربة مرشح 26 أكتوبر');root.mainloop()

if __name__=='__main__':main()
