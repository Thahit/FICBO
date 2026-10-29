import argparse
import os
import time
from functools import partial
from typing import Any, Dict, List, Optional, Tuple
import torch
from expert_models import ModelExpert, SklearnExpertModel, build_sklearn_model
from tasks import BenchmarkFeedbackTask
from utils.bench_eval import (
    AveragedPolicyModel, MethodConfig, baseline_configs, evaluate_gp_bo, evaluate_policy_vs_bo,
    load_config_and_model, pi_bo, run_benchmark_evaluation, seed_everything,
    )
from utils.bench_summary import write_summary


#just a cheap generated port to use cluster
HERE = os.path.dirname(os.path.abspath(__file__))

# (hydra run dir relative to --outputs-root, checkpoint file)
# example paths, replace with your own training runs
MODEL_PATHS: Dict[str, Tuple[str, str]] = {
    "combined": ("outputs/hydra/mixed_prior/YYYY-MM-DD/HH-MM-SS/", "model/ficbo_combined.pth"),
    "mixture": ("outputs/hydra/ficbo_prior/YYYY-MM-DD/HH-MM-SS/", "model/ficbo_mixture.pth"),
    "additive": ("outputs/hydra/additive_prior/YYYY-MM-DD/HH-MM-SS/", "model/ficbo_additive.pth"),
    "gp_icbo": ("outputs/hydra/gp_icbo/YYYY-MM-DD/HH-MM-SS/", "model/gp_icbo.pth"),
    "gp_icbo_fb": ("outputs/hydra/gp_icbo_feedback_feature/YYYY-MM-DD/HH-MM-SS/",
                         "model/gp_icbo_feedback_feature.pth"),
}

BENCHMARKS: List[str] = [
    "sphere3dn", "ackley3dn", "rastrigin3dn", "rosenbrock3dn", "schwefel3dn", "levy3dn", "hartmann3dn",
    "sphere2dn", "ackley2dn", "rastrigin2dn", "rosenbrock2dn", "schwefel2dn", "levy2dn",
]

FAKE_PEAK = dict(fake_peak_k=1, fake_peak_height_mult=1.1, fake_peak_decay_range=(0.5, 0.5))
PRESETS: Dict[str, Dict[str, Any]] = {
    "marginalized": dict(n_marginal_samples=2000, fake_peaks=False, expert="none"),
    "xgboost": dict(n_marginal_samples=1, fake_peaks=False, expert="sklearn"),
    "fakepeak": dict(n_marginal_samples=2000, fake_peaks=True, expert="none"),
    "fakepeak_nomarg": dict(n_marginal_samples=1, fake_peaks=True, expert="none"),
}

N_ITER_GP = 100
N_MTGP_ITER = 30
MTGP_ACQUISITIONS = ("ucb",)
MTGP_KAPPA = 1.0
EXPERT_DIM = 1
DESIGN_SCALE = 5
NOISE_SCALE = 0.1
RGPE_ACQUISITIONS = ("ucb",)
RGPE_MAX_POOL = 300
RGPE_NUM_POSTERIOR_SAMPLES = 256
IMFBO_ACQUISITIONS = ("nvucb", "ucb")
IMFBO_NOISE_MODELS = ("parametric", "gp")
IMFBO_MAX_POOL = 300
N_IMFBO_ITER = 50
N_QUERY_INIT = 300
EXPERT_KWARGS = dict(
    n_expert_extra_min=50, n_expert_extra_max=50,
    expert_min_centers=3, expert_max_centers=3,
    expert_cluster_std_min=0.5, expert_cluster_std_max=1.5,
    uniform_fraction=1.0,
    )

# run settings that must match for --resume to append to an existing raw.pt
RUN_KEYS = ("n_marginal_samples", "fake_peaks", "expert", "seed", "batch_size", "T", "fake_peak_k",
            "fake_peak_height_mult", "fake_peak_decay_range")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="All synthetic-benchmark stats in one call. Main eval is notebooks/eval_with_expert.ipynb.")
    p.add_argument("--preset", choices=sorted(PRESETS), default="marginalized")
    p.add_argument("--n-marginal-samples", type=int, default=None, help="override the preset")
    p.add_argument("--fake-peaks", action=argparse.BooleanOptionalAction, default=None,
                   help="override the preset")
    p.add_argument("--expert", choices=["none", "sklearn"], default=None,
                   help="feedback generator, sklearn = cfg.expert_model.sklearn_model (xgboost); override the preset")
    p.add_argument("--seed", type=int, default=42, help="reseeded per benchmark")
    p.add_argument("--batch-size", type=int, default=100, help="environments per benchmark")
    p.add_argument("--T", type=int, default=30)
    p.add_argument("--benchmarks", nargs="+", default=BENCHMARKS)
    p.add_argument("--only", nargs="+", default=None, help="restrict to these method keys")
    p.add_argument("--no-mtgp", action="store_true")
    p.add_argument("--no-rgpe", action="store_true")
    p.add_argument("--no-imfbo", action="store_true")
    p.add_argument("--reference", default="ficbo_mixture", help="method for paired deltas")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--outputs-root", default=".",
                   help="directory the MODEL_PATHS are relative to")
    p.add_argument("--out-dir", default=os.path.join(HERE, "notebooks", "bench_results"))
    p.add_argument("--from-raw", default=None, help="only re-summarize this raw.pt")
    return p.parse_args()


def run_settings(args: argparse.Namespace) -> Dict[str, Any]:
    s = dict(PRESETS[args.preset], **FAKE_PEAK)
    if args.n_marginal_samples is not None:
        s["n_marginal_samples"] = args.n_marginal_samples
    if args.fake_peaks is not None:
        s["fake_peaks"] = args.fake_peaks
    if args.expert is not None:
        s["expert"] = args.expert
    s.update(preset=args.preset, seed=args.seed, batch_size=args.batch_size, T=args.T)
    return s


def run_dir_name(s: Dict[str, Any]) -> str:
    name = s["preset"]
    if s["n_marginal_samples"] != PRESETS[s["preset"]]["n_marginal_samples"]:
        name += f"_marg{s['n_marginal_samples']}"
    if s["fake_peaks"] != PRESETS[s["preset"]]["fake_peaks"]:
        name += "_peaks" if s["fake_peaks"] else "_nopeaks"
    if s["expert"] != PRESETS[s["preset"]]["expert"]:
        name += f"_expert{s['expert']}"
    if s["seed"] != 42:
        name += f"_seed{s['seed']}"
    if (s["batch_size"], s["T"]) != (100, 30):
        name += f"_B{s['batch_size']}_T{s['T']}"
    return name


def load_models(args: argparse.Namespace) -> Dict[str, Tuple[Any, Any]]:
    models = {}
    for key, (rel, file_name) in MODEL_PATHS.items():
        path = os.path.join(args.outputs_root, rel)
        if not os.path.isdir(path):
            raise FileNotFoundError(f"{path} not found, set MODEL_PATHS['{key}'] to your training run")
        cfg, model = load_config_and_model(path=path, file_name=file_name)
        models[key] = (cfg, model.to(args.device))

    # only the main model's cfg is adjusted
    cfg, model = models["combined"]
    model.embedder.embedding_type = "mix"
    cfg.task.embedding_type = "mix"
    cfg.task.mask_type = ["none"]
    cfg.task.n_query_init = 400
    return models


def build_method_configs(
    models: Dict[str, Tuple[Any, Any]], args: argparse.Namespace
    ) -> List[MethodConfig]:
    cfg, model = models["combined"]
    cfg_mixture, model_mixture = models["mixture"]
    cfg_additive, model_additive = models["additive"]
    cfg_no_fb, model_no_fb = models["gp_icbo"]
    cfg_fb, model_fb = models["gp_icbo_fb"]
    model_ensemble = AveragedPolicyModel(model_mixture, model_additive)
    n_theta = cfg.task.n_target_theta
    n_theta_baselines = cfg_no_fb.task.n_target_theta
    pibo = partial(pi_bo, beta=2.)

    def amortized(policy_model: Any, adjustor: Optional[Any]) -> MethodConfig:
        return MethodConfig(
            name="_", model=policy_model, cfg=cfg, attend_to="data", n_theta=n_theta_baselines,
            use_gp=False, gp_model=None, expert_feedback_adjustor=adjustor, advanced_stats=False,
            concat_feedback=False, n_iter_gp=N_ITER_GP,
            )

    def torch_gp(adjustor: Optional[Any], concat_feedback: bool = False) -> MethodConfig:
        return MethodConfig(
            name="_", model=model_no_fb, cfg=cfg, attend_to="data", n_theta=n_theta_baselines,
            use_gp=True, gp_model=None, expert_feedback_adjustor=adjustor, advanced_stats=False,
            concat_feedback=concat_feedback, n_iter_gp=N_ITER_GP, kappa=MTGP_KAPPA,
            )

    def mtgp(kind: str) -> MethodConfig:
        return MethodConfig(
            name="_", model=model_no_fb, cfg=cfg, attend_to="data", n_theta=n_theta_baselines,
            use_multitask_gp=True, mtgp_kind=kind, expert_feedback_adjustor=None,
            advanced_stats=False, n_iter_gp=N_MTGP_ITER, kappa=MTGP_KAPPA,
            )

    def rgpe() -> MethodConfig:
        return MethodConfig(
            name="_", model=model_no_fb, cfg=cfg, attend_to="data", n_theta=n_theta_baselines,
            use_rgpe=True, rgpe_max_pool=RGPE_MAX_POOL,
            rgpe_num_posterior_samples=RGPE_NUM_POSTERIOR_SAMPLES,
            rgpe_target_noise_var=NOISE_SCALE ** 2, rgpe_feedback_noise_var=NOISE_SCALE ** 2,
            expert_feedback_adjustor=None, advanced_stats=False, kappa=MTGP_KAPPA,
            )

    def imfbo(noise_model: str) -> MethodConfig:
        return MethodConfig(
            name="_", model=model_no_fb, cfg=cfg, attend_to="data", n_theta=n_theta_baselines,
            use_imfbo=True, imfbo_noise_model=noise_model, imfbo_max_pool=IMFBO_MAX_POOL,
            expert_feedback_adjustor=None, advanced_stats=False, n_iter_gp=N_IMFBO_ITER,
            kappa=MTGP_KAPPA,
            )

    configs = [
        MethodConfig("ficbo_combined", model, cfg, acquisition="aae", attend_to="data", n_theta=n_theta),
        MethodConfig("ficbo_mixture", model_mixture, cfg_mixture, acquisition="aae",
                     attend_to="data", n_theta=n_theta),
        MethodConfig("ficbo_additive", model_additive, cfg_additive, acquisition="aae",
                     attend_to="data", n_theta=n_theta),
        MethodConfig("ficbo_ensemble", model_ensemble, cfg_mixture, acquisition="aae",
                     attend_to="data", n_theta=n_theta),
        MethodConfig("gp_icbo", model_no_fb, cfg, acquisition="aae", attend_to="data",
                     n_theta=n_theta_baselines, concat_feedback=False),
        MethodConfig("gp_icbo_fb", model_fb, cfg, acquisition="aae", attend_to="data",
                     n_theta=cfg_fb.task.n_target_theta, concat_feedback=True,
                     experiment_dim_x_delta=cfg.task.dim_feedback),
        *baseline_configs(amortized(model_no_fb, None), acquisitions=("random", "feedback")),
        ]
    if not args.no_mtgp:
        configs += baseline_configs(mtgp("icm"), acquisitions=MTGP_ACQUISITIONS, name_prefix="mtgp_icm_")
    if not args.no_imfbo:
        for nm in IMFBO_NOISE_MODELS:
            prefix = "imfbo_" if nm == "parametric" else f"imfbo_{nm}_"
            configs += baseline_configs(imfbo(nm), acquisitions=IMFBO_ACQUISITIONS, name_prefix=prefix)
    if not args.no_rgpe:
        configs += baseline_configs(rgpe(), acquisitions=RGPE_ACQUISITIONS, name_prefix="rgpe_")
    configs += [
        *baseline_configs(torch_gp(None, concat_feedback=False), acquisitions=MTGP_ACQUISITIONS,
                          name_prefix="gp_"),
        *baseline_configs(torch_gp(None, concat_feedback=True), acquisitions=MTGP_ACQUISITIONS,
                          name_prefix="gp_feat_"),
        *baseline_configs(amortized(model_no_fb, pibo), acquisitions=("ucb", "ei"), name_prefix="pibo_"),
        *baseline_configs(amortized(model_mixture, pibo), acquisitions=("ucb", "ei"),
                          name_prefix="pibo_mixture_"),
        *baseline_configs(torch_gp(pibo), acquisitions=("ucb", "ei"), name_prefix="gp_pibo_"),
        ]

    if args.only is not None:
        unknown = set(args.only) - {mc.name for mc in configs}
        if unknown:
            raise ValueError(f"--only: unknown method keys {sorted(unknown)}")
        configs = [mc for mc in configs if mc.name in args.only]
    return configs


def save_raw(path: str, raw: Dict[str, Any]) -> None:
    # write-then-rename so a job killed mid-save never leaves a truncated raw.pt behind
    tmp = path + ".tmp"
    torch.save(raw, tmp)
    os.replace(tmp, path)


def main() -> None:
    args = parse_args()

    if args.from_raw is not None:
        torch.set_num_threads(1)
        raw = torch.load(args.from_raw, map_location="cpu", weights_only=False)
        write_summary(raw["results"], raw["optima"], os.path.dirname(os.path.abspath(args.from_raw)),
                      reference=args.reference)
        return

    settings = run_settings(args)
    out_dir = os.path.join(args.out_dir, run_dir_name(settings))
    os.makedirs(out_dir, exist_ok=True)
    raw_path = os.path.join(out_dir, "raw.pt")
    print(f"Run settings: {settings}\nOutput: {out_dir}")

    raw = {"results": {}, "optima": {}, "settings": settings, "method_names": None}
    if os.path.exists(raw_path):
        prev = torch.load(raw_path, map_location="cpu", weights_only=False)
        prev["settings"].setdefault("expert", "none")  # raw.pt from before --expert existed
        mismatch = {k: (prev["settings"].get(k), settings.get(k)) for k in RUN_KEYS
                    if prev["settings"].get(k) != settings.get(k)}
        if mismatch:
            raise RuntimeError(f"{raw_path} was made with different settings {mismatch}; move it away")
        raw = prev
        print(f"Resuming: {list(raw['results'])} already done")

    models = load_models(args)
    method_configs = build_method_configs(models, args)
    names = [mc.name for mc in method_configs]
    if raw["method_names"] not in (None, names):
        raise RuntimeError(f"{raw_path} has methods {raw['method_names']}, this run has {names}")
    raw["method_names"] = names
    cfg = models["combined"][0]
    expert_generator = None
    if settings["expert"] == "sklearn":
        expert_generator = partial(
            ModelExpert, model=SklearnExpertModel(build_sklearn_model(cfg.expert_model.sklearn_model)),
            fit_on=["expert"], n_jobs=int(os.environ.get("SLURM_CPUS_PER_TASK", 1)), adaptive=False,
            normalize_outputs_to_max_one=False, mode="mean",
            )

    for bench in args.benchmarks:
        if bench in raw["results"]:
            continue
        t0 = time.time()
        # One call per benchmark: run_benchmark_evaluation reseeds at the start of every
        # benchmark, so this gives the same numbers as one call over the whole list.
        results, optima = run_benchmark_evaluation(
            benchmarks=[bench],
            method_configs=method_configs,
            experiment=None,  # unused by run_benchmark_evaluation
            evaluate_policy_vs_bo=evaluate_policy_vs_bo,
            evaluate_gp_bo=evaluate_gp_bo,
            BenchmarkFeedbackTask=BenchmarkFeedbackTask,
            cfg=cfg,
            batch_size=settings["batch_size"],
            T=settings["T"],
            expert_dim=EXPERT_DIM,
            n_marginal_samples=settings["n_marginal_samples"],
            design_scale=DESIGN_SCALE,
            noise_scale=NOISE_SCALE,
            expert_bias_model=None,
            expert_generator=expert_generator,
            fake_peaks=settings["fake_peaks"],
            fake_peak_k=settings["fake_peak_k"],
            fake_peak_height_mult=settings["fake_peak_height_mult"],
            fake_peak_decay_range=settings["fake_peak_decay_range"],
            device=args.device,
            seed_fn=partial(seed_everything, settings["seed"]),
            verbose=1,
            n_query_init=N_QUERY_INIT,
            **EXPERT_KWARGS,
            )
        raw["results"][bench] = results[bench]
        raw["optima"][bench] = optima[bench]
        save_raw(raw_path, raw)
        print(f"[{bench}] done in {time.time() - t0:.0f}s, saved {raw_path}")

    write_summary(raw["results"], raw["optima"], out_dir, reference=args.reference)


if __name__ == "__main__":
    main()
