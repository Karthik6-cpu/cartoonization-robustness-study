
from pathlib import Path
import subprocess, json, time, sys

SRC = Path("/content/cartoonization-robustness-study/research/CAFRNet_v2/source")
DATA = Path("/content/cafr_data")
BASE = Path("/content/cafr_runs/imagenette")
JOBS = [
    ("cafrnet_v2",123), ("cafrnet_v2",314),
    ("convnext_tiny",42), ("convnext_tiny",123), ("convnext_tiny",314),
    ("resnet18",42), ("resnet18",123), ("resnet18",314)
]

for i, (arch, seed) in enumerate(JOBS, 1):
    out = BASE / f"{arch}_seed{seed}"
    out.mkdir(parents=True, exist_ok=True)
    latest = out / "latest_model.pt"
    best = out / "best_model.pt"
    log = out / "training.log"

    if latest.exists():
        import torch
        ck = torch.load(latest, map_location="cpu", weights_only=False)
        epoch = int(ck.get("epoch", 0))
        if epoch >= 20 and best.exists():
            print(f"[{i}/{len(JOBS)}] SKIP completed {arch} seed {seed}", flush=True)
            continue
        resume = epoch > 0
    else:
        resume = False

    cmd = [
        sys.executable, str(SRC/"train_architecture.py"),
        "--arch", arch,
        "--data", str(DATA),
        "--out", str(out),
        "--seed", str(seed),
        "--epochs", "20",
        "--batch-size", "16",
        "--workers", "2",
        "--image-size", "224",
        "--encoder-lr", "2e-5",
        "--head-lr", "2e-4",
        "--weight-decay", "0.05",
        "--freeze-epochs", "2",
        "--lambda-consistency", "0.25",
        "--ablation", "full"
    ]
    if resume:
        cmd.append("--resume")

    print(f"\n[{i}/{len(JOBS)}] START {arch} seed {seed}; resume={resume}", flush=True)
    with log.open("a", encoding="utf-8") as f:
        f.write("\\nCOMMAND: " + " ".join(cmd) + "\\n")
        result = subprocess.run(cmd, cwd=SRC, stdout=f, stderr=subprocess.STDOUT)
    if result.returncode != 0:
        print(f"FAILED: {arch} seed {seed}; inspect {log}", flush=True)
        raise SystemExit(result.returncode)
    print(f"COMPLETED: {arch} seed {seed}", flush=True)

print("\\nALL ARCHITECTURE RUNS FINISHED", flush=True)
