"""Train CAFR-Net and preserve resumable artifacts after every epoch.

Expected folder layout:
  DATA_ROOT/train/<class>/*.(jpg|jpeg|png)
  DATA_ROOT/val/<class>/*.(jpg|jpeg|png)

This is an initial paired-view robustness training pipeline. It currently
uses JPEG/blur/brightness/noise views, not AnimeGAN or Stable Diffusion.
Do not describe it as full cartoonization training until those train-only
pipelines are integrated and verified.
"""
from __future__ import annotations
import argparse, io, json, random, zipfile, hashlib, platform, csv
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
import torchvision

from models.cafr_net import CAFRNet, paired_consistency_loss

MEAN=(0.485,0.456,0.406); STD=(0.229,0.224,0.225)

def seed_everything(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def sha256_file(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""): h.update(chunk)
    return h.hexdigest()

def shifted_view(image):
    kind=random.choice(("jpeg","blur","brightness","noise","identity"))
    if kind=="blur": return image.filter(ImageFilter.GaussianBlur(random.uniform(.3,1.2)))
    if kind=="brightness": return ImageEnhance.Brightness(image).enhance(random.uniform(.75,1.25))
    if kind=="jpeg":
        b=io.BytesIO(); image.save(b,format="JPEG",quality=random.choice((35,50,65,80)))
        b.seek(0); return Image.open(b).convert("RGB")
    if kind=="noise":
        arr=np.asarray(image).astype(np.float32)
        arr+=np.random.normal(0,random.uniform(3,12),arr.shape)
        return Image.fromarray(np.uint8(np.clip(arr,0,255)))
    return image.copy()

class PairedImageFolder(Dataset):
    def __init__(self,root,image_size=224):
        self.base=ImageFolder(root)
        self.geometry=transforms.Compose([transforms.Resize((image_size,image_size)),
                                          transforms.RandomHorizontalFlip(.5)])
        self.tensor=transforms.Compose([transforms.ToTensor(),transforms.Normalize(MEAN,STD)])
    def __len__(self): return len(self.base)
    def __getitem__(self,index):
        path,label=self.base.samples[index]
        image=self.base.loader(path).convert("RGB")
        clean=self.geometry(image); shifted=shifted_view(clean)
        return self.tensor(clean),self.tensor(shifted),label

@torch.no_grad()
def validate(model,loader,device):
    model.eval(); correct=total=0; loss_sum=0.
    ce=nn.CrossEntropyLoss()
    for images,labels in loader:
        images,labels=images.to(device),labels.to(device)
        logits=model(images); loss=ce(logits,labels)
        loss_sum+=loss.item()*labels.size(0)
        correct+=(logits.argmax(1)==labels).sum().item(); total+=labels.size(0)
    return {"val_loss":loss_sum/max(total,1),"val_accuracy":correct/max(total,1)}

def make_zip(run_dir,zip_path):
    zip_path=Path(zip_path); zip_path.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(zip_path,"w",zipfile.ZIP_DEFLATED) as z:
        for p in Path(run_dir).rglob("*"):
            if p.is_file() and p.resolve()!=zip_path.resolve():
                z.write(p,p.relative_to(run_dir))
    print("Artifact ZIP:",zip_path,flush=True)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--data",required=True)
    p.add_argument("--out",default="/content/cafr_runs/seed42")
    p.add_argument("--epochs",type=int,default=20)
    p.add_argument("--batch-size",type=int,default=32)
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--workers",type=int,default=2)
    p.add_argument("--lambda-consistency",type=float,default=.25)
    p.add_argument("--image-size",type=int,default=224)
    p.add_argument("--resume",action="store_true")
    args=p.parse_args()
    seed_everything(args.seed)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root=Path(args.data); out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    if not (root/"train").is_dir() or not (root/"val").is_dir():
        raise FileNotFoundError("Expected DATA_ROOT/train and DATA_ROOT/val folders.")
    train_ds=PairedImageFolder(str(root/"train"),args.image_size)
    val_ds=ImageFolder(str(root/"val"),transforms.Compose([
        transforms.Resize((args.image_size,args.image_size)),transforms.ToTensor(),
        transforms.Normalize(MEAN,STD)]))
    if train_ds.base.classes!=val_ds.classes: raise ValueError("Train/val class names differ.")
    train_loader=DataLoader(train_ds,batch_size=args.batch_size,shuffle=True,
        num_workers=args.workers,pin_memory=device.type=="cuda",
        persistent_workers=args.workers>0)
    val_loader=DataLoader(val_ds,batch_size=args.batch_size,shuffle=False,
        num_workers=args.workers,pin_memory=device.type=="cuda",
        persistent_workers=args.workers>0)
    model=CAFRNet(num_classes=len(train_ds.base.classes),pretrained=True).to(device)
    for q in model.encoder.parameters(): q.requires_grad=False
    optimizer=torch.optim.AdamW([
        {"params":model.encoder.parameters(),"lr":2e-5},
        {"params":[q for n,q in model.named_parameters() if not n.startswith("encoder.")],"lr":2e-4}
    ],weight_decay=.05)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=args.epochs)
    scaler=torch.amp.GradScaler("cuda",enabled=device.type=="cuda")
    start_epoch=0; best_acc=-1.; stale=0; history=[]
    best_path=out/"best_cafr_net.pt"; latest_path=out/"latest_cafr_net.pt"
    if args.resume and latest_path.exists():
        ck=torch.load(latest_path,map_location=device,weights_only=False)
        model.load_state_dict(ck["model"]); optimizer.load_state_dict(ck["optimizer"])
        scheduler.load_state_dict(ck["scheduler"])
        start_epoch=ck["epoch"]; best_acc=ck.get("best_accuracy",-1.)
        history=ck.get("history",[])
        print("Resuming at epoch",start_epoch+1)
    meta={"created_utc":datetime.now(timezone.utc).isoformat(),"seed":args.seed,
          "device":str(device),"gpu":torch.cuda.get_device_name(0) if device.type=="cuda" else None,
          "torch":torch.__version__,"torchvision":torchvision.__version__,
          "python":platform.python_version(),"classes":train_ds.base.classes,
          "train_samples":len(train_ds),"validation_samples":len(val_ds),
          "train_root":str(root/"train"),"validation_root":str(root/"val"),
          "args":vars(args)}
    (out/"run_metadata.json").write_text(json.dumps(meta,indent=2))
    ce=nn.CrossEntropyLoss()
    try:
        for epoch in range(start_epoch,args.epochs):
            if epoch==2:
                for q in model.encoder.parameters(): q.requires_grad=True
            model.train(); running=total=0.
            for clean,shifted,labels in train_loader:
                clean,shifted,labels=clean.to(device),shifted.to(device),labels.to(device)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type,enabled=device.type=="cuda"):
                    a=model(clean); b=model(shifted)
                    loss=ce(a,labels)+ce(b,labels)+args.lambda_consistency*paired_consistency_loss(a,b)
                scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
                running+=loss.item()*labels.size(0); total+=labels.size(0)
            metrics=validate(model,val_loader,device); scheduler.step()
            row={"epoch":epoch+1,"train_loss":running/max(total,1),**metrics}
            history.append(row)
            with open(out/"history.jsonl","a",encoding="utf-8") as f:
                f.write(json.dumps(row)+"\n")
            print(json.dumps(row),flush=True)
            if metrics["val_accuracy"]>best_acc:
                best_acc=metrics["val_accuracy"]; stale=0
                torch.save({"model":model.state_dict(),"classes":train_ds.base.classes,
                    "seed":args.seed,"epoch":epoch+1,"val_accuracy":best_acc,"args":vars(args)},best_path)
            else: stale+=1
            torch.save({"model":model.state_dict(),"optimizer":optimizer.state_dict(),
                "scheduler":scheduler.state_dict(),"epoch":epoch+1,"best_accuracy":best_acc,
                "history":history,"classes":train_ds.base.classes,"args":vars(args)},latest_path)
            make_zip(out,out.parent/(out.name+"_artifacts.zip"))
            if stale>=5: print("Early stopping."); break
    finally:
        (out/"history.json").write_text(json.dumps(history,indent=2))
        make_zip(out,out.parent/(out.name+"_artifacts.zip"))
    print("Best validation accuracy:",best_acc)
    print("Best checkpoint:",best_path)

if __name__=="__main__": main()
