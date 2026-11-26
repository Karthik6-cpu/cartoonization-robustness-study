"""Evaluate a matched architecture checkpoint on identical ImageFolder conditions.

Example:
python evaluate_architecture.py --checkpoint /content/cafr_runs/convnext_seed42/best_model.pt \
  --conditions original=/content/cafr_test opencv=/content/cafr_opencv \
  --out /content/cafr_runs/convnext_seed42/predictions.csv
"""
import argparse, csv
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.models import convnext_tiny, resnet18

from cafr_net import CAFRNetV2

MEAN=(0.485,0.456,0.406)
STD=(0.229,0.224,0.225)


def make_model(arch,num_classes,model_config):
    if arch=="cafrnet_v2":
        return CAFRNetV2(num_classes=num_classes,pretrained=False,**model_config)
    if arch=="convnext_tiny":
        model=convnext_tiny(weights=None)
        model.classifier[2]=nn.Linear(model.classifier[2].in_features,num_classes)
        return model
    if arch=="resnet18":
        model=resnet18(weights=None)
        model.fc=nn.Linear(model.fc.in_features,num_classes)
        return model
    raise ValueError(f"Unknown architecture in checkpoint: {arch}")


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--checkpoint",required=True)
    ap.add_argument("--conditions",nargs="+",required=True,help="NAME=PATH pairs")
    ap.add_argument("--out",required=True)
    ap.add_argument("--batch-size",type=int,default=64)
    ap.add_argument("--workers",type=int,default=2)
    ap.add_argument("--image-size",type=int,default=224)
    args=ap.parse_args()

    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt=torch.load(args.checkpoint,map_location="cpu",weights_only=False)
    arch=ckpt["architecture"]
    classes=ckpt["classes"]
    model=make_model(arch,len(classes),ckpt.get("model_config",{}))
    model.load_state_dict(ckpt["model"],strict=True)
    model.to(device).eval()
    tf=transforms.Compose([
        transforms.Resize((args.image_size,args.image_size)),
        transforms.ToTensor(),transforms.Normalize(MEAN,STD)
    ])
    rows=[]
    with torch.inference_mode():
        for spec in args.conditions:
            name,folder=spec.split("=",1)
            folder=Path(folder)
            ds=ImageFolder(str(folder),transform=tf)
            if ds.classes!=classes:
                raise ValueError(
                    f"Class order mismatch for {name}: expected {classes}, got {ds.classes}"
                )
            loader=DataLoader(
                ds,batch_size=args.batch_size,shuffle=False,
                num_workers=args.workers,pin_memory=device.type=="cuda"
            )
            offset=0
            for images,labels in loader:
                logits=model(images.to(device))
                probs=logits.softmax(1).cpu()
                conf,pred=probs.max(1)
                for j in range(len(labels)):
                    full_path=Path(ds.samples[offset+j][0])
                    rel=str(full_path.relative_to(folder))
                    y=int(labels[j]); p=int(pred[j])
                    rows.append({
                        "architecture":arch,
                        "seed":ckpt.get("seed","unknown"),
                        "condition":name,
                        "relative_path":rel,
                        "true_label":classes[y],
                        "predicted_label":classes[p],
                        "confidence":float(conf[j]),
                        "correct":int(p==y)
                    })
                offset+=len(labels)
    out=Path(args.out); out.parent.mkdir(parents=True,exist_ok=True)
    fields=["architecture","seed","condition","relative_path","true_label",
            "predicted_label","confidence","correct"]
    with out.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=fields)
        w.writeheader(); w.writerows(rows)
    print(f"Saved {len(rows)} image-condition predictions to {out}")


if __name__=="__main__":
    main()
