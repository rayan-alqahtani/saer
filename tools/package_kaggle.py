"""تحزيم خادم Gradio والأصول المعتمدة دون تدريب أو رفع."""
from pathlib import Path
import hashlib,json,zipfile
ROOT=Path(__file__).resolve().parents[1]
PRIVATE=["models/image_price_current.json","models/image_price_hamoud_20261028/bundle.joblib","models/improvement_current.json","models/improvement_20261021T093545513846/price/best_model.joblib","models/identity_current.json","outputs/generalization_rescue_20261028_v1/model/candidate.joblib"]
def main():
    out=ROOT/"build/kaggle/saer-backend-bundle.zip"
    out.parent.mkdir(parents=True,exist_ok=True)
    files=[]
    for rel in PRIVATE:
        p=ROOT/rel
        if not p.is_file(): raise FileNotFoundError(rel)
        data=p.read_bytes()
        files.append({"path":rel,"size_bytes":len(data),"sha256":hashlib.sha256(data).hexdigest()})
    manifest={"files":files,"total_bytes":sum(x["size_bytes"] for x in files)}
    with zipfile.ZipFile(out,"w",zipfile.ZIP_DEFLATED) as archive:
        for p in sorted((ROOT/"hf_backend").rglob("*")):
            if p.is_file() and "__pycache__" not in p.parts and p.suffix in {".py",".txt",".md"}:
                archive.write(p,p.relative_to(ROOT/"hf_backend").as_posix())
        for rel in PRIVATE: archive.write(ROOT/rel,rel)
        archive.writestr("private_assets_manifest.json",json.dumps(manifest,ensure_ascii=False,indent=2))
    print(out)
    print("Private assets:",len(files),"bytes:",manifest["total_bytes"])
if __name__=="__main__":main()
