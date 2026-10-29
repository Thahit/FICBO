import math
from typing import Dict, List, Optional, Tuple

import pandas as pd
import torch


# results key -> display name, mathtext for plots and tex. list every eval_benchmarks.py method
METHOD_LABELS: Dict[str, str] = {
    # FICBO
    "ficbo_mixture": "FICBO (Mixture Prior)",
    "ficbo_additive": "FICBO (Additive Prior)",
    "ficbo_combined": "FICBO (Combined)",
    "ficbo_ensemble": "FICBO (Ensemble)",
    # GP-prior amortized baselines
    "gp_icbo_fb": "GP-prior ICBO + feedback feature",
    "gp_icbo": "GP-prior ICBO",
    # piBO
    "pibo_ucb": r"$\pi$BO-UCB",
    "pibo_ei": r"$\pi$BO-EI",
    "gp_pibo_ucb": r"$\pi$BO-UCB GP",
    "gp_pibo_ei": r"$\pi$BO-EI GP",
    # Multi-task / transfer baselines
    "mtgp_icm_ucb": "MTGP-UCB (ICM)",
    "rgpe_ucb": "RGPE-UCB",
    "imfbo_nvucb": "iMFBO-NVUCB",
    "imfbo_ucb": "iMFBO-UCB",
    "imfbo_gp_nvucb": "iMFBO-GP-NVUCB",
    "imfbo_gp_ucb": "iMFBO-GP-UCB",
    # Single-task GP controls
    "gp_ucb": "GP-UCB",
    "gp_feat_ucb": "GP-UCB + feedback feature",
    # Simple baselines
    "feedback": "Feedback",
    "random": "Random",
    # Extra
    "pibo_mixture_ucb": r"$\pi$BO-UCB (Mixture-prior surrogate)",
    "pibo_mixture_ei": r"$\pi$BO-EI (Mixture-prior surrogate)",
}

# raw.pt from before the rename to paper names
LEGACY_KEYS: Dict[str, str] = {
    "ours_Reweighted": "ficbo_mixture", "ours_Amortized": "ficbo_additive", "ours": "ficbo_combined",
    "ours_Combined": "ficbo_ensemble", "aline_with_feedback": "gp_icbo_fb", "aline": "gp_icbo",
    "pibo_OurPost_ucb": "pibo_mixture_ucb", "pibo_OurPost_ei": "pibo_mixture_ei",
}
LEGACY_DROPPED = {"mbo_hf"}  # not in paper

# metric -> higher is better
METRICS: Dict[str, bool] = {
    "Best Found": True,
    "Norm. Best": True,
    "Regret": False,
    "Cum. Regret": False,
}

ALL_ROW = "ALL (mean)"


def _per_env_metrics(res: tuple, opt: torch.Tensor) -> Dict[str, torch.Tensor]:
    # per env [B], for std / sem / paired deltas later
    best = res[1].float().cpu()  # [T+1, B]
    init = best[0:1]
    norm = (best - init) / (opt.unsqueeze(0) - init + 1e-8)
    return {
        "Best Found": best[-1],
        "Norm. Best": norm[-1],
        "Regret": opt - best[-1],
        "Cum. Regret": (1.0 - norm[1:]).sum(dim=0),
        }


def _mean_std_sem(x: torch.Tensor) -> Tuple[float, float, float]:
    x = x.double().flatten()
    n = x.numel()
    if n < 2:
        return x.mean().item(), 0.0, 0.0
    std = x.std(unbiased=True).item()
    return x.mean().item(), std, std / math.sqrt(n)


def _check_shared_start(bench: str, per_method: Dict[str, tuple]) -> None:
    # paired deltas need same start, warn only
    starts = {k: v[1][0].float().cpu() for k, v in per_method.items()}
    ref = next(iter(starts.values()))
    bad = [k for k, s in starts.items() if not torch.allclose(s, ref)]
    if bad:
        print(f"[summary] WARNING {bench}: t=0 best differs for {bad}; paired deltas are not paired")


def per_benchmark_table(
    results: Dict[str, Dict[str, Optional[tuple]]],
    optima: Dict[str, torch.Tensor],
    labels: Dict[str, str] = METHOD_LABELS,
    reference: Optional[str] = "ficbo_mixture",
    ) -> pd.DataFrame:
    """
    One row per (benchmark, method) with final-step performance, plus an ALL row per method.

    Args:
        results: results[bench][method] = run_benchmark_evaluation tuple (None if it failed).
        optima: optima[bench] = [B] pool maximum per environment.
        labels: results key -> display name.
        reference: method the paired "vs ref" columns are taken against (None to skip).

    Returns:
        DataFrame with, for every metric M: "M" (mean), "M std", "M sem", and (if reference is
        set) "M vs ref", "M vs ref sem". Per-benchmark rows use the B environments. ALL rows use
        the mean of per-benchmark means, sem = sqrt(sum sem_k^2)/K, std = RMS of per-benchmark
        stds, and "M std across benches" = spread of the per-benchmark means.
    """
    results = {b: {LEGACY_KEYS.get(k, k): v for k, v in per.items() if k not in LEGACY_DROPPED}
               for b, per in results.items()}
    rows = []
    for bench, per_method_all in results.items():
        failed = [k for k, v in per_method_all.items() if v is None]
        if failed:
            print(f"[summary] WARNING {bench}: no results (method failed) for {failed}")
        per_method = {k: v for k, v in per_method_all.items() if v is not None}
        if not per_method:
            continue
        _check_shared_start(bench, per_method)

        opt = optima[bench].float().cpu()
        env = {k: _per_env_metrics(v, opt) for k, v in per_method.items()}
        ref_env = env.get(reference) if reference is not None else None
        if reference is not None and ref_env is None:
            print(f"[summary] WARNING {bench}: reference '{reference}' missing, no paired deltas")

        for key, metrics in env.items():
            if key not in labels:
                print(f"[summary] WARNING: no label for '{key}', using the key")
            row = {"Benchmark": bench, "Key": key, "Method": labels.get(key, key)}
            for m, vals in metrics.items():
                row[m], row[f"{m} std"], row[f"{m} sem"] = _mean_std_sem(vals)
                if ref_env is not None:
                    d_mean, _, d_sem = _mean_std_sem(vals - ref_env[m])
                    row[f"{m} vs ref"], row[f"{m} vs ref sem"] = d_mean, d_sem
            row["n"] = next(iter(metrics.values())).numel()
            rows.append(row)

    df = pd.DataFrame(rows)
    df["Rank"] = df.groupby("Benchmark")["Norm. Best"].rank(ascending=False, method="min")

    all_rows = []
    for key, g in df.groupby("Key", sort=False):
        k = len(g)
        row = {"Benchmark": ALL_ROW, "Key": key, "Method": g["Method"].iloc[0], "n": int(g["n"].sum())}
        for m in METRICS:
            row[m] = g[m].mean()
            row[f"{m} sem"] = math.sqrt((g[f"{m} sem"] ** 2).sum()) / k
            row[f"{m} std"] = math.sqrt((g[f"{m} std"] ** 2).mean())
            row[f"{m} std across benches"] = g[m].std(ddof=1) if k > 1 else 0.0
            if f"{m} vs ref" in g:
                row[f"{m} vs ref"] = g[f"{m} vs ref"].mean()
                row[f"{m} vs ref sem"] = math.sqrt((g[f"{m} vs ref sem"] ** 2).sum()) / k
        row["Mean Rank"] = g["Rank"].mean()
        row["Mean Rank sem"] = g["Rank"].std(ddof=1) / math.sqrt(k) if k > 1 else 0.0
        row["n benches"] = k
        all_rows.append(row)
    all_df = pd.DataFrame(all_rows)
    all_df["Rank"] = all_df["Norm. Best"].rank(ascending=False, method="min")

    df["Mean Rank"] = df["Rank"]
    df["Mean Rank sem"] = 0.0
    df["n benches"] = 1
    df = pd.concat([df, all_df], ignore_index=True)
    df["Rank"] = df["Rank"].astype(int)

    order = list(results.keys()) + [ALL_ROW]
    df["Benchmark"] = pd.Categorical(df["Benchmark"], categories=order, ordered=True)
    df = df.sort_values(["Benchmark", "Rank"]).reset_index(drop=True)

    lead = ["Benchmark", "Method", "Key", "Rank", "Mean Rank", "Mean Rank sem", "n", "n benches"]
    return df[lead + [c for c in df.columns if c not in lead]]


def format_table(df: pd.DataFrame, which: str = "sem") -> pd.DataFrame:
    """
    Collapse mean/uncertainty pairs into latex 'mean pm unc' strings, indexed by (Benchmark, Method).

    Args:
        df: output of per_benchmark_table.
        which: "sem" or "std" -- which uncertainty column to show next to each mean.
    """
    out = pd.DataFrame(index=pd.MultiIndex.from_frame(df[["Benchmark", "Method"]]))
    for col, fmt in [("Best Found", "{:.3f}"), ("Norm. Best", "{:.3f}"),
                     ("Regret", "{:.3f}"), ("Cum. Regret", "{:.2f}")]:
        out[col] = [rf"{fmt.format(m)} $\pm$ {fmt.format(s)}" for m, s in zip(df[col], df[f"{col} {which}"])]
    out["Rank"] = df["Rank"].values
    return out


def print_overall(df: pd.DataFrame, reference: Optional[str] = "ficbo_mixture") -> None:
    all_df = df[df["Benchmark"] == ALL_ROW].sort_values("Rank")
    has_delta = "Norm. Best vs ref" in all_df
    print(f"\n{ALL_ROW} over {int(all_df['n benches'].max())} benchmarks (+- sem)")
    header = f"  {'#':>2}  {'Method':<42} {'Norm. Best':>17} {'Cum. Regret':>17} {'Mean Rank':>14}"
    if has_delta:
        header += f" {'Norm.Best vs ' + str(reference):>28}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for _, r in all_df.iterrows():
        line = (
            f"  {r['Rank']:>2}  {r['Method']:<42} "
            f"{r['Norm. Best']:>8.3f} +- {r['Norm. Best sem']:<6.3f} "
            f"{r['Cum. Regret']:>8.2f} +- {r['Cum. Regret sem']:<6.2f} "
            f"{r['Mean Rank']:>5.1f} +- {r['Mean Rank sem']:<5.1f}"
            )
        if has_delta and not pd.isna(r.get("Norm. Best vs ref")):
            line += f" {r['Norm. Best vs ref']:>+17.3f} +- {r['Norm. Best vs ref sem']:<6.3f}"
        print(line)


def write_summary(
    results: Dict[str, Dict[str, Optional[tuple]]], optima: Dict[str, torch.Tensor], out_dir: str,
    reference: Optional[str] = "ficbo_mixture",
    ) -> pd.DataFrame:
    df = per_benchmark_table(results, optima, reference=reference)
    df.to_csv(f"{out_dir}/summary.csv", index=False)
    for which in ("sem", "std"):
        with open(f"{out_dir}/summary_{which}.tex", "w") as f:
            f.write(format_table(df, which).to_latex(escape=False))
    print_overall(df, reference)
    print(f"\nWrote {out_dir}/summary.csv, summary_sem.tex, summary_std.tex")
    return df

