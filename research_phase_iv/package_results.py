"""Package experiment artifacts for download and later GitHub upload."""
import argparse, json, zipfile, hashlib
from pathlib import Path
from datetime import datetime, timezone

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--input",required=True,help="Experiment output directory")
    p.add_argument("--zip",default="/content/CAFRNet_results.zip")
    p.add_argument("--include-data",action="store_true",
                   help="Normally leave datasets out; include only if licensing permits.")
    a=p.parse_args()
    src=Path(a.input).resolve(); dest=Path(a.zip).resolve()
    if not src.exists(): raise FileNotFoundError(src)
    dest.parent.mkdir(parents=True,exist_ok=True)
    manifest=[]
    with zipfile.ZipFile(dest,"w",zipfile.ZIP_DEFLATED) as z:
        for f in src.rglob("*"):
            if not f.is_file(): continue
            if not a.include_data and any(part.lower() in {"train","val","test","dataset","datasets","images"} for part in f.parts):
                continue
            arc=f.relative_to(src)
            z.write(f,arc)
            h=hashlib.sha256(f.read_bytes()).hexdigest()
            manifest.append({"path":str(arc),"bytes":f.stat().st_size,"sha256":h})
        z.writestr("BUNDLE_MANIFEST.json",json.dumps({
            "created_utc":datetime.now(timezone.utc).isoformat(),
            "source_directory":str(src),"files":manifest},indent=2))
    print("Created:",dest)
    print("Files:",len(manifest),"ZIP bytes:",dest.stat().st_size)

if __name__=="__main__": main()
