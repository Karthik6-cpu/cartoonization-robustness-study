"""Evaluate a CAFR-Net checkpoint on one or more condition folders.

Each condition folder must use ImageFolder layout:
  CONDITION_ROOT/<class_name>/<image files>

The CSV preserves relative image paths and per-image predictions so matched
source IDs can be joined to the repository's verified manifest downstream.
"""
import argparse, csv
from pathlib import Path

import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

from models.cafr_net import CAFRNet

MEAN=(0.485,0.456,0.406); STD=(0.229,0.224,0.225)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--conditions", nargs="+", required=True,
                   help="NAME=PATH pairs")
    p.add_argument("--out", default="cafr_predictions.csv")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--workers", type=int, default=4)
    a=p.parse_args()
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt=torch.load(a.checkpoint,map_location="cpu",weights_only=False)
    classes=ckpt["classes"]
    model=CAFRNet(num_classes=len(classes),pretrained=False)
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    tf=transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(),
                           transforms.Normalize(MEAN,STD)])
    rows=[]
    with torch.no_grad():
        for item in a.conditions:
            name,path=item.split("=",1)
            ds=ImageFolder(path,transform=tf)
            if ds.classes != classes:
                raise ValueError(f"Class mismatch in condition {name}")
            loader=DataLoader(ds,batch_size=a.batch_size,shuffle=False,
                              num_workers=a.workers)
            offset=0
            for images,labels in loader:
                logits=model(images.to(device))
                probs=logits.softmax(1).cpu()
                conf,pred=probs.max(1)
                for j in range(len(labels)):
                    rel=ds.samples[offset+j][0]
                    rows.append({
                        "condition":name,
                        "relative_path":str(Path(rel).relative_to(path)),
                        "true_label":classes[int(labels[j])],
                        "predicted_label":classes[int(pred[j])],
                        "confidence":float(conf[j]),
                        "correct":int(pred[j])==int(labels[j]),
                    })
                offset += len(labels)
    with open(a.out,"w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0].keys()))
        writer.writeheader(); writer.writerows(rows)
    print(f"Wrote {len(rows)} predictions to {a.out}")


if __name__=="__main__":
    main()
