"""حزمة مراجعة مختصرة للنماذج المختارة دون وسطاء activations الكبيرة."""
from improve_common import *
import subprocess,time,shutil


def build():
    guard();s=session();folder=s/'release';folder.mkdir(exist_ok=False)
    files=[]
    for task in ['make','model','color','body','price','search']:
        cfg=load_json(s/task/'selected.json');dest=folder/task;dest.mkdir()
        for name in ['selected.json','val_metrics.json','test_metrics.json','test_per_class.csv','test_worst10.csv','test_once.json']:
            p=s/task/name
            if p.exists():shutil.copy2(p,dest/name)
        if cfg.get('kind')=='tail':shutil.copy2(s/cfg['checkpoint'],dest/'model.pt')
        if task in TASK_INFO:shutil.copy2(original(task)/'model.joblib',dest/'reference_classifier.joblib')
        if task=='make' and (s/task/'adapted_head.joblib').exists():shutil.copy2(s/task/'adapted_head.joblib',dest/'adapted_head.joblib')
        if task=='price':shutil.copy2(s/task/'best_model.joblib',dest/'model.joblib')
    shutil.copy2(original('embeddings')/'encoder_weights.pth',folder/'reference_encoder_weights.pth')
    shutil.copy2(s/'experiments.csv',folder/'experiments.csv');shutil.copy2(s/'protocol.json',folder/'protocol.json')
    shutil.copy2(ROOT/'requirements-models.txt',folder/'requirements.txt')
    # تشغيل من نفس المشروع مع تثبيت الجلسة، حتى إذا تغير المؤشر لاحقًا.
    launcher='''import os,sys\nfrom pathlib import Path\nhere=Path(__file__).resolve().parent\nos.environ['CAR_IMPROVEMENT_RUN']=str(here.parent)\nsys.path.insert(0,str(here.parents[2]/'scripts'))\nfrom improve_predict import main\nraise SystemExit(main())\n'''
    save_bytes(folder/'predict.py',launcher.encode('utf-8'))
    row=compact_rows('price','val')[0];save_json(folder/'مثال_سعر.json',{k:row[k] for k in ['make','model_std','year','km_mileage','trim_std','city','transmission','type_fuel','condition']})
    guide='''<div dir="rtl" lang="ar">

# تشغيل اختيارات التحسين المحلية

هذه الحزمة المختصرة للمراجعة تشمل أوزان الاختيارات والمراجع المحفوظة والمقاييس والتجارب. لا تنقل activation_cache للتسليم؛ هي وسطاء تدريب كبيرة قابلة لإعادة الحساب وليست صورًا أصلية أو شرطًا للاستدلال الخارجي.

المشغل الحالي محلي داخل مشروع C:/CAR ويحتاج بقاء كود المشروع والجلسة والنماذج المرجعية، لا يدعي أنه مثبت مستقل لجهاز آخر. لم تستبدل الحزمة النماذج الأصلية. يثبت predict.py هذه الجلسة تحديدًا. اختر النموذج على val فقط؛ الانخفاض في test لا يطلق إعادة اختيار تلقائية.

من PowerShell داخل هذا المجلد:

```powershell
& 'C:/CAR/.venv/Scripts/python.exe' -X utf8 ./predict.py status
& 'C:/CAR/.venv/Scripts/python.exe' -X utf8 ./predict.py price --input './مثال_سعر.json'
& 'C:/CAR/.venv/Scripts/python.exe' -X utf8 ./predict.py make --input 'C:/مسار/سيارة.jpg'
& 'C:/CAR/.venv/Scripts/python.exe' -X utf8 ./predict.py model --input 'C:/مسار/سيارة.jpg'
& 'C:/CAR/.venv/Scripts/python.exe' -X utf8 ./predict.py color --input 'C:/مسار/سيارة.jpg'
& 'C:/CAR/.venv/Scripts/python.exe' -X utf8 ./predict.py body --input 'C:/مسار/سيارة.jpg'
& 'C:/CAR/.venv/Scripts/python.exe' -X utf8 ./predict.py search --input 'C:/مسار/سيارة.jpg' --k 5
```

تظهر اقتراحات منخفضة الثقة كغير متأكد. درجات القرار ليست وعدًا بالصحة خارج المصدر. النوع body سيدان أو SUV ونحو ذلك، وليس كشف ضرر أو سلامة شاصي. تقدير السعر سعر إعلان للمستعمل، وليس سعر بيع أو تقدير خسارة حادث. لم تختبر الصور الخارجية الحقيقية بعد؛ ضع 20–30 صورة بوسوم يدوية في external_test ثم شغل سكربت التقييم المخصص.

نتائج قبول التشغيل في qa.json، والتقرير العربي في C:/CAR/docs/MODEL_IMPROVEMENT_2026-10-21.md. ملفات الاختبار الجديدة تسجل حدثًا نهائيًا واحدًا لكل مهمة؛ لا تعاد لتجربة معاملات بعد ظهور نتائجها.

</div>
'''
    save_bytes(folder/'دليل_التشغيل.md',guide.encode('utf-8'))
    save_json(folder/'files_sha256.json',{str(p.relative_to(folder)):sha256(p) for p in folder.rglob('*') if p.is_file()})
    print('RELEASE',folder,'MB',round(sum(p.stat().st_size for p in folder.rglob('*') if p.is_file())/1e6,1),flush=True)


def verify():
    s=session();folder=s/'release';checks=[];py=ROOT/'.venv/Scripts/python.exe'
    expected=load_json(folder/'files_sha256.json')
    checks.append({'check':'package_hashes','passed':all(sha256(folder/name)==digest for name,digest in expected.items())})
    photo=ROOT/'الملفات_المهمة'/compact_rows('model','val')[0]['relative_path']
    bad=s/'invalid_image.txt';save_bytes(bad,'ملف ليس صورة'.encode('utf-8'))
    for task in ['status','price','make','model','color','body','search','invalid_image']:
        output=s/'qa'/f'{task}.json';command=[str(py),'-X','utf8',str(folder/'predict.py'),'model' if task=='invalid_image' else task,'--output',str(output)]
        if task!='status':command.extend(['--input',str(folder/'مثال_سعر.json' if task=='price' else bad if task=='invalid_image' else photo)])
        start=time.perf_counter();p=subprocess.run(command,cwd=ROOT/'tmp',capture_output=True,text=True,encoding='utf-8',timeout=180)
        payload=load_json(output,{})
        try:console=json.loads(p.stdout)==payload
        except (ValueError,TypeError):console=False
        passed=p.returncode==(1 if task=='invalid_image' else 0) and payload.get('نجاح')==(task!='invalid_image') and console
        checks.append({'check':task,'passed':passed,'seconds':time.perf_counter()-start,'stderr':p.stderr[-1500:]})
        print('FRESH PROCESS',task,passed,flush=True)
    guard();checks.append({'check':'original_models_unchanged','passed':True})
    result={'checked_at':now(),'passed':all(c['passed'] for c in checks),'checks':checks,'scope':'fresh process on val photo only, no extra test evaluation'}
    save_json(folder/'qa.json',result)
    if not result['passed']:raise SystemExit(1)


if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument('command',choices=['build','verify']);a=ap.parse_args()
    if a.command=='build':build()
    else:verify()
