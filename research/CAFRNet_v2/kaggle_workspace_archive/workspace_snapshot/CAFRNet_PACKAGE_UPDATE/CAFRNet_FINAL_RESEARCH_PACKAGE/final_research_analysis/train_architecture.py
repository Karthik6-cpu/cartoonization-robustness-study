"""Matched training harness for CAFR-Net v2 and plain torchvision baselines.
Run from this directory (source/):
  python train_architecture.py --arch convnext_tiny --data /content/cafr_data --out 
/content/cafr_runs/convnext_seed42 --seed 42 --epochs 20
  python train_architecture.py --arch resnet18 --data /content/cafr_data --out 
/content/cafr_runs/resnet_seed42 --seed 42 --epochs 20
  python train_architecture.py --arch cafrnet_v2 --data /content/cafr_data --out 
/content/cafr_runs/cafr_seed42 --seed 42 --epochs 20
All three architectures use the same paired-view dataset, augmentations, optimizer schedule,
loss, split, image size and training budget. This makes the plain ConvNeXt-Tiny comparison
the primary backbone-control experiment.
"""
from __future__ import annotations
import argparse, csv, hashlib, io, json, random, platform
from pathlib import Path
from datetime import datetime, timezone
import cv2
import numpy as np
import torch
import torchvision
from PIL import Image, ImageEnhance, ImageFilter
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.models import (
    convnext_tiny, ConvNeXt_Tiny_Weights,
    resnet18, ResNet18_Weights,
)
from cafr_net import CAFRNetV2, paired_consistency_loss
MEAN=(0.485,0.456,0.406)
STD=(0.229,0.224,0.225)
def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)
def shifted_view(image):
    kind = random.choices(
        ("opencv_cartoon", "jpeg", "blur", "brightness", "noise", "identity"),
        weights=(0.35, 0.15, 0.15, 0.10, 0.10, 0.15), k=1
    )[0]
    if kind == "opencv_cartoon":
        arr=np.asarray(image.convert("RGB"))
        bgr=cv2.cvtColor(arr,cv2.COLOR_RGB2BGR)
        out=cv2.stylization(bgr,sigma_s=60,sigma_r=0.45)
        return Image.fromarray(cv2.cvtColor(out,cv2.COLOR_BGR2RGB))
    if kind == "jpeg":
        b=io.BytesIO()
        image.save(b,format="JPEG",quality=random.choice((35,50,65,80)))
        b.seek(0)
        return Image.open(b).convert("RGB")
    if kind == "blur":
        return image.filter(ImageFilter.GaussianBlur(random.uniform(.3,1.2)))
    if kind == "brightness":
        return ImageEnhance.Brightness(image).enhance(random.uniform(.75,1.25))
    if kind == "noise":
        arr=np.asarray(image).astype(np.float32)
        arr += np.random.normal(0,random.uniform(3,12),arr.shape)
        return Image.fromarray(np.uint8(np.clip(arr,0,255)))
    return image.copy()
class PairedImageFolder(Dataset):
    def __init__(self,root,image_size=224):
        self.base=ImageFolder(str(root))
        self.geometry=transforms.Compose([
            transforms.Resize((image_size,image_size)),
            transforms.RandomHorizontalFlip(.5)
        ])
        self.tensor=transforms.Compose([
            transforms.ToTensor(),transforms.Normalize(MEAN,STD)
        ])
    def __len__(self):
        return len(self.base)
    def __getitem__(self,index):
        path,label=self.base.samples[index]
        image=self.base.loader(path).convert("RGB")
        clean=self.geometry(image)
        shifted=shifted_view(clean)
        return self.tensor(clean),self.tensor(shifted),label
def build_model(arch,num_classes,ablation="full"):
    if arch=="cafrnet_v2":
        cfg={
            "use_edge":ablation!="no_edge",
            "use_semantic_fusion":ablation!="no_semantic_fusion",
            "use_attention":ablation!="no_attention",
        }
        model=CAFRNetV2(num_classes=num_classes,pretrained=False,**cfg)
        encoder=model.encoder
        head_params=[p for n,p in model.named_parameters() if not n.startswith("encoder.")]
        return model, list(encoder.parameters()), head_params, cfg
    if ablation!="full":
        raise ValueError("Ablations are defined only for cafrnet_v2.")
    if arch=="convnext_tiny":
        model=convnext_tiny(weights=ConvNeXt_Tiny_Weights.DEFAULT)
        in_features=model.classifier[2].in_features
        model.classifier[2]=nn.Linear(in_features,num_classes)
        encoder=model.features
        head_params=list(model.classifier.parameters())
        return model,list(encoder.parameters()),head_params,{}
    if arch=="resnet18":
        model=resnet18(weights=ResNet18_Weights.DEFAULT)
        in_features=model.fc.in_features
        model.fc=nn.Linear(in_features,num_classes)
        encoder=nn.Sequential(
            model.conv1,model.bn1,model.relu,model.maxpool,
            model.layer1,model.layer2,model.layer3,model.layer4,model.avgpool
        )
        head_params=list(model.fc.parameters())
        encoder_params=[p for n,p in model.named_parameters() if not n.startswith("fc.")]
        return model,encoder_params,head_params,{}
    raise ValueError(f"Unknown architecture: {arch}")
@torch.no_grad()
def validate(model,loader,device):
    model.eval()
    correct=total=0
    loss_sum=0.0
    ce=nn.CrossEntropyLoss()
    for images,labels in loader:
        images,labels=images.to(device),labels.to(device)
        logits=model(images)
        loss=ce(logits,labels)
        loss_sum += loss.item()*labels.size(0)
        correct += (logits.argmax(1)==labels).sum().item()
        total += labels.size(0)
    return {"val_loss":loss_sum/max(total,1),"val_accuracy":correct/max(total,1)}
def write_split_manifest(dataset,output):
    rows=[]
    for path,label in dataset.base.samples:
        rows.append((str(Path(path).relative_to(dataset.base.root)),dataset.base.classes[label]))
    rows.sort()
    raw="\n".join(f"{p}\t{y}" for p,y in rows).encode()
    digest=hashlib.sha256(raw).hexdigest()
    with open(output,"w",newline="",encoding="utf-8") as f:
        w=csv.writer(f); w.writerow(["relative_path","class_name"]); w.writerows(rows)
    return digest
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--arch",choices=("cafrnet_v2","convnext_tiny","resnet18"),required=True)
    ap.add_argument("--data",required=True,help="Root containing train/ and val/ ImageFolder directories")
    ap.add_argument("--out",required=True)
    ap.add_argument("--epochs",type=int,default=20)
    ap.add_argument("--batch-size",type=int,default=32)
    ap.add_argument("--seed",type=int,required=True)
    ap.add_argument("--workers",type=int,default=2)
    ap.add_argument("--image-size",type=int,default=224)
    ap.add_argument("--lambda-consistency",type=float,default=.25)
    ap.add_argument("--encoder-lr",type=float,default=2e-5)
    ap.add_argument("--head-lr",type=float,default=2e-4)
    ap.add_argument("--weight-decay",type=float,default=.05)
    ap.add_argument("--freeze-epochs",type=int,default=2)
    ap.add_argument("--resume",action="store_true",help="Resume from OUT/latest_model.pt")
    ap.add_argument("--ablation",choices=
("full","no_edge","no_semantic_fusion","no_attention"),default="full")
    args=ap.parse_args()
    seed_everything(args.seed)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root=Path(args.data); out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    if not (root/"train").is_dir() or not (root/"val").is_dir():
        raise FileNotFoundError("Expected DATA_ROOT/train and DATA_ROOT/val.")
    train_ds=PairedImageFolder(root/"train",args.image_size)
    val_ds=ImageFolder(str(root/"val"),transforms.Compose([
        transforms.Resize((args.image_size,args.image_size)),
        transforms.ToTensor(),transforms.Normalize(MEAN,STD)
    ]))
    if train_ds.base.classes!=val_ds.classes:
        raise ValueError("Train and validation class names/order differ.")
    generator=torch.Generator(); generator.manual_seed(args.seed)
    train_loader=DataLoader(
        train_ds,batch_size=args.batch_size,shuffle=True,num_workers=args.workers,
        pin_memory=device.type=="cuda",persistent_workers=args.workers>0,
        worker_init_fn=seed_worker,generator=generator
    )
    val_loader=DataLoader(
        val_ds,batch_size=args.batch_size,shuffle=False,num_workers=args.workers,
        pin_memory=device.type=="cuda",persistent_workers=args.workers>0
    )
    model,encoder_params,head_params,model_config=build_model(
        args.arch,len(train_ds.base.classes),args.ablation
    )
    model=model.to(device)
    for p in encoder_params: p.requires_grad=False
    optimizer=torch.optim.AdamW([
        {"params":encoder_params,"lr":args.encoder_lr},
        {"params":head_params,"lr":args.head_lr}
    ],weight_decay=args.weight_decay)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=args.epochs)
    scaler=torch.amp.GradScaler("cuda",enabled=device.type=="cuda")
    split_hash=write_split_manifest(train_ds,out/"train_split_manifest.csv")
    val_rows=sorted(
        (str(Path(p).relative_to(val_ds.root)),val_ds.classes[y])
        for p,y in val_ds.samples
    )
    with open(out/"validation_split_manifest.csv","w",newline="",encoding="utf-8") as f:
        w=csv.writer(f); w.writerow(["relative_path","class_name"]); w.writerows(val_rows)
    val_hash=hashlib.sha256("\n".join(f"{p}\t{y}" for p,y in val_rows).encode()).hexdigest()
    metadata={
        "created_utc":datetime.now(timezone.utc).isoformat(),
        "architecture":args.arch,"ablation":args.ablation,"seed":args.seed,
        "device":str(device),"gpu":torch.cuda.get_device_name(0) if device.type=="cuda" else None,
        "python":platform.python_version(),"torch":torch.__version__,
        "torchvision":torchvision.__version__,"classes":train_ds.base.classes,
        "train_samples":len(train_ds),"validation_samples":len(val_ds),
        "parameter_count_total":sum(p.numel() for p in model.parameters()),
        "parameter_count_trainable_at_start":sum(p.numel() for p in model.parameters() if 
p.requires_grad),
        "train_manifest_sha256":split_hash,"validation_manifest_sha256":val_hash,
        "training_protocol":{
            "epochs":args.epochs,"batch_size":args.batch_size,"image_size":args.image_size,
            "optimizer":"AdamW","encoder_lr":args.encoder_lr,"head_lr":args.head_lr,
            "weight_decay":args.weight_decay,"scheduler":"CosineAnnealingLR",
            "freeze_epochs":args.freeze_epochs,"paired_views":True,
            "lambda_consistency":args.lambda_consistency,
            "augmentation_probabilities":{
                "opencv_cartoon":.35,"jpeg":.15,"blur":.15,
                "brightness":.10,"gaussian_noise":.10,"identity":.15
            }
        },
        "arguments":vars(args)
    }
    (out/"run_metadata.json").write_text(json.dumps(metadata,indent=2))
    ce=nn.CrossEntropyLoss()
    best_acc=-1.0
    history=[]
    start_epoch=0
    latest_path=out/"latest_model.pt"
    if args.resume:
        if not latest_path.exists():
            raise FileNotFoundError(f"No resumable checkpoint found: {latest_path}")
        ck=torch.load(latest_path,map_location=device,weights_only=False)
        if ck.get("architecture")!=args.arch or ck.get("seed")!=args.seed:
            raise ValueError("Resume checkpoint architecture/seed does not match this run.")
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        scheduler.load_state_dict(ck["scheduler"])
        scaler.load_state_dict(ck.get("scaler",{}))
        start_epoch=int(ck["epoch"])
        best_acc=float(ck.get("best_accuracy",-1.0))
        history=ck.get("history",[])
        if "loader_generator_state" in ck:
            generator.set_state(
                ck["loader_generator_state"].detach().cpu().to(dtype=torch.uint8)
            )
        if "python_rng_state" in ck:
            random.setstate(ck["python_rng_state"])
        if "numpy_rng_state" in ck:
            np.random.set_state(ck["numpy_rng_state"])
        if "torch_rng_state" in ck:
            torch.set_rng_state(
                ck["torch_rng_state"].detach().cpu().to(dtype=torch.uint8)
            )
        if torch.cuda.is_available() and ck.get("cuda_rng_state"):
            torch.cuda.set_rng_state_all([
                s.detach().cpu().to(dtype=torch.uint8)
                for s in ck["cuda_rng_state"]
            ])
        if start_epoch>=args.freeze_epochs:
            for p in encoder_params: p.requires_grad=True
        print(f"Resuming {args.arch} seed {args.seed} at epoch {start_epoch+1}",flush=True)
    for epoch in range(start_epoch,args.epochs):
        if epoch==args.freeze_epochs:
            for p in encoder_params: p.requires_grad=True
        model.train()
        if epoch<args.freeze_epochs:
            # Keep frozen encoder statistics fixed (important for ResNet BatchNorm).
            if args.arch=="resnet18":
                model.eval()
                model.fc.train()
            else:
                encoder_module=model.encoder if args.arch=="cafrnet_v2" else model.features
                encoder_module.eval()
                if args.arch=="cafrnet_v2":
                    model.projections.train(); model.fusion.train()
                    model.edge_refinement.train(); model.attention.train()
                    model.refinement.train(); model.classifier.train()
                else:
                    model.classifier.train()
        running=total=0.0
        for clean,shifted,labels in train_loader:
            clean,shifted,labels=clean.to(device),shifted.to(device),labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type,enabled=device.type=="cuda"):
                logits_a=model(clean)
                logits_b=model(shifted)
                loss=ce(logits_a,labels)+ce(logits_b,labels)
                loss=loss+args.lambda_consistency*paired_consistency_loss(logits_a,logits_b)
            scaler.scale(loss).backward()
            scaler.step(optimizer); scaler.update()
            running+=loss.item()*labels.size(0); total+=labels.size(0)
        metrics=validate(model,val_loader,device)
        scheduler.step()
        row={"epoch":epoch+1,"train_loss":running/max(total,1),**metrics}
        history.append(row)
        print(json.dumps(row),flush=True)
        with open(out/"history.jsonl","a",encoding="utf-8") as f:
            f.write(json.dumps(row)+"\n")
        if metrics["val_accuracy"]>best_acc:
            best_acc=metrics["val_accuracy"]
            torch.save({
                "model":model.state_dict(),"architecture":args.arch,
                "model_config":model_config,"classes":train_ds.base.classes,
                "seed":args.seed,"epoch":epoch+1,"val_accuracy":best_acc,
                "metadata":metadata
            },out/"best_model.pt")
        torch.save({
            "model":model.state_dict(),"architecture":args.arch,
            "model_config":model_config,"classes":train_ds.base.classes,
            "seed":args.seed,"epoch":epoch+1,"best_accuracy":best_acc,
            "optimizer":optimizer.state_dict(),"scheduler":scheduler.state_dict(),
            "scaler":scaler.state_dict(),"history":history,"metadata":metadata,
            "loader_generator_state":generator.get_state(),
            "python_rng_state":random.getstate(),"numpy_rng_state":np.random.get_state(),
            "torch_rng_state":torch.get_rng_state(),
            "cuda_rng_state":torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        },latest_path)
    (out/"history.json").write_text(json.dumps(history,indent=2))
    print("Best validation accuracy:",best_acc)
    print("Best checkpoint:",out/"best_model.pt")
if __name__=="__main__":
    main()
