#!/usr/bin/env python3
"""Generate deterministic, matched compound-corruption images from a TEST-ONLY
class-folder dataset. Input: root/class_name/image.ext. Does not train models.
"""
from pathlib import Path
from PIL import Image, ImageFilter, ImageEnhance
import argparse, csv, hashlib, io, random

def stable_seed(rel, base):
    return int(hashlib.sha256(f"{base}:{rel}".encode()).hexdigest()[:8], 16)

def gaussian_noise(im, rng, sigma=12.0):
    im = im.convert("RGB")
    px = im.load()
    for y in range(im.height):
        for x in range(im.width):
            r,g,b = px[x,y]
            px[x,y] = tuple(max(0,min(255,int(v+rng.gauss(0,sigma)))) for v in (r,g,b))
    return im

def jpeg_roundtrip(im, quality=50):
    buf = io.BytesIO()
    im.convert("RGB").save(buf, format="JPEG", quality=quality, optimize=True)
    buf.seek(0)
    return Image.open(buf).convert("RGB").copy()

def make_conditions(im, rng):
    clean = im.convert("RGB")
    noise = gaussian_noise(clean, rng)
    blur = clean.filter(ImageFilter.GaussianBlur(radius=1.2))
    bright = ImageEnhance.Brightness(clean).enhance(1.25)
    noise_blur = gaussian_noise(clean.filter(ImageFilter.GaussianBlur(radius=1.2)), rng)
    jpeg_noise = gaussian_noise(jpeg_roundtrip(clean, 50), rng)
    jpeg_blur = jpeg_roundtrip(clean.filter(ImageFilter.GaussianBlur(radius=1.2)), 50)
    return {
        "clean": clean,
        "jpeg_q50": jpeg_roundtrip(clean, 50),
        "gaussian_noise": noise,
        "gaussian_blur": blur,
        "brightness_up": bright,
        "noise_plus_blur": noise_blur,
        "jpeg_q50_plus_noise": jpeg_noise,
        "jpeg_q50_plus_blur": jpeg_blur,
    }

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="TEST split only: root/class_name/images")
    ap.add_argument("--output", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-images", type=int, default=0, help="0 means all; use 100-300 for a quick pilot")
    args=ap.parse_args()
    src=Path(args.input); dst=Path(args.output)
    if not src.is_dir(): raise SystemExit(f"Input folder not found: {src}")
    files=sorted(p for p in src.rglob("*") if p.suffix.lower() in {".jpg",".jpeg",".png",".webp"})
    if args.max_images: files=files[:args.max_images]
    if not files: raise SystemExit("No images found. Expected class-folder structure.")
    rows=[]
    for idx,path in enumerate(files,1):
        rel=path.relative_to(src)
        label=rel.parts[0] if len(rel.parts)>1 else "UNSPECIFIED"
        with Image.open(path) as im:
            im=im.convert("RGB")
            conditions=make_conditions(im, random.Random(stable_seed(str(rel),args.seed)))
            for cond,out in conditions.items():
                target=dst/cond/rel.with_suffix(".png")
                target.parent.mkdir(parents=True,exist_ok=True)
                out.save(target,format="PNG")
                rows.append([str(rel),label,cond,str(target),args.seed])
        if idx%100==0: print(f"Processed {idx}/{len(files)} source images")
    dst.mkdir(parents=True,exist_ok=True)
    with (dst/"matched_corruption_manifest.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.writer(f); w.writerow(["source_id","label","condition","output_path","seed"]); w.writerows(rows)
    print(f"Done: {len(files)} source images, {len(rows)} matched condition records.")
    print(f"Manifest: {dst/'matched_corruption_manifest.csv'}")
    print("Reminder: run inference using the exact same source IDs for every condition.")

if __name__=="__main__": main()
