"""Aggregate matched architecture predictions and run paired comparisons.

Input is one concatenated CSV produced by evaluate_architecture.py, with rows
from every architecture/seed/condition. All models must use identical test IDs.
Outputs are per-seed metrics, mean/SD across seeds, and paired exact McNemar
tests with Holm correction. Bootstrap intervals are conditional on each seed's
checkpoints and must not be mistaken for training-seed uncertainty.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest, t
from sklearn.metrics import accuracy_score, f1_score


def holm_adjust(p_values):
    p=np.asarray(p_values,dtype=float)
    m=len(p)
    order=np.argsort(p)
    adjusted=np.empty(m,dtype=float)
    running=0.0
    for rank,idx in enumerate(order):
        value=min(1.0,(m-rank)*p[idx])
        running=max(running,value)
        adjusted[idx]=running
    return adjusted


def paired_bootstrap_delta(a,b,reps=2000,seed=2026):
    # delta = accuracy(A) - accuracy(B), using the same sampled image indices.
    rng=np.random.default_rng(seed)
    n=len(a)
    deltas=np.empty(reps,dtype=float)
    a=np.asarray(a,dtype=int); b=np.asarray(b,dtype=int)
    for i in range(reps):
        ix=rng.integers(0,n,n)
        deltas[i]=a[ix].mean()-b[ix].mean()
    return float(np.quantile(deltas,.025)),float(np.quantile(deltas,.975))


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--predictions",required=True)
    ap.add_argument("--outdir",required=True)
    ap.add_argument("--reference",default="cafrnet_v2")
    ap.add_argument("--bootstrap",type=int,default=2000)
    args=ap.parse_args()

    df=pd.read_csv(args.predictions)
    required={"architecture","seed","condition","relative_path","true_label","predicted_label","correct"}
    missing=required-set(df.columns)
    if missing: raise ValueError(f"Missing columns: {sorted(missing)}")
    if df.duplicated(["architecture","seed","condition","relative_path"]).any():
        raise ValueError("Duplicate prediction identity found.")
    df["correct"]=df["correct"].astype(int)
    out=Path(args.outdir); out.mkdir(parents=True,exist_ok=True)

    metric_rows=[]
    for (arch,seed,condition),g in df.groupby(["architecture","seed","condition"]):
        metric_rows.append({
            "architecture":arch,"seed":seed,"condition":condition,"n":len(g),
            "accuracy":accuracy_score(g.true_label,g.predicted_label),
            "macro_f1":f1_score(g.true_label,g.predicted_label,average="macro",zero_division=0)
        })
    per_seed=pd.DataFrame(metric_rows)
    per_seed.to_csv(out/"per_seed_metrics.csv",index=False)

    mean_rows=[]
    for (arch,condition),g in per_seed.groupby(["architecture","condition"]):
        for metric in ["accuracy","macro_f1"]:
            vals=g[metric].to_numpy(dtype=float)
            mean=float(vals.mean()); sd=float(vals.std(ddof=1)) if len(vals)>1 else float("nan")
            if len(vals)>1:
                sem=sd/np.sqrt(len(vals))
                half=float(t.ppf(.975,len(vals)-1)*sem)
                lo,hi=mean-half,mean+half
            else: lo=hi=float("nan")
            mean_rows.append({
                "architecture":arch,"condition":condition,"metric":metric,
                "n_seeds":len(vals),"mean":mean,"sd":sd,
                "ci95_seed_level_low":lo,"ci95_seed_level_high":hi
            })
    pd.DataFrame(mean_rows).to_csv(out/"mean_sd_across_seeds.csv",index=False)

    archs=sorted(df.architecture.unique())
    if args.reference not in archs:
        raise ValueError(f"Reference architecture {args.reference} not found: {archs}")
    comparisons=[]
    for other in archs:
        if other==args.reference: continue
        for (seed,condition),refg in df[df.architecture==args.reference].groupby(["seed","condition"]):
            othg=df[(df.architecture==other)&(df.seed==seed)&(df.condition==condition)]
            keys=["relative_path","true_label"]
            merged=refg.merge(othg,on=keys,suffixes=("_ref","_other"),validate="one_to_one")
            if len(merged)!=len(refg) or len(merged)!=len(othg):
                raise ValueError(f"Unmatched images for {seed}/{condition}/{other}")
            a=merged["correct_ref"].to_numpy(dtype=int)
            b=merged["correct_other"].to_numpy(dtype=int)
            ref_only=int(((a==1)&(b==0)).sum())
            other_only=int(((a==0)&(b==1)).sum())
            discordant=ref_only+other_only
            pval=1.0 if discordant==0 else float(binomtest(min(ref_only,other_only),discordant,.5,alternative="two-sided").pvalue)
            lo,hi=paired_bootstrap_delta(a,b,args.bootstrap,seed=int(seed)+len(condition))
            comparisons.append({
                "reference":args.reference,"other":other,"seed":seed,"condition":condition,
                "n":len(merged),"reference_accuracy":float(a.mean()),
                "other_accuracy":float(b.mean()),"difference_pp":float((a.mean()-b.mean())*100),
                "reference_only_correct":ref_only,"other_only_correct":other_only,
                "mcnemar_exact_p":pval,
                "bootstrap_delta_pp_low":lo*100,"bootstrap_delta_pp_high":hi*100
            })
    if comparisons:
        p_adj=holm_adjust([x["mcnemar_exact_p"] for x in comparisons])
        for row,p in zip(comparisons,p_adj): row["mcnemar_holm_p"]=float(p)
    pd.DataFrame(comparisons).to_csv(out/"paired_model_comparisons.csv",index=False)
    print("Wrote:")
    print(out/"per_seed_metrics.csv")
    print(out/"mean_sd_across_seeds.csv")
    print(out/"paired_model_comparisons.csv")


if __name__=="__main__":
    main()
