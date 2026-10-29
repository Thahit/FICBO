# FICBO: Feedback-Informed In-Context Black-Box Optimization

FICBO is a feedback-aware amortized Bayesian optimizer. 
A pretrained transformer conditions on the optimization history and on cheap auxiliary feedback (experts, simulators, heuristics), and infers in context how much to trust that feedback.

> Nicolas Samuel Blumer, Julien Martinelli, Samuel Kaski.
> *In-Context Black-Box Optimization with Unreliable Feedback.* NeurIPS 2026.
> arXiv:2605.06187

The code builds on [ALINE](https://github.com/huangdaolang/ALINE) (Huang et al., NeurIPS 2025).

## Installation

```bash
conda create -n ficbo python=3.12
conda activate ficbo
pip install -r requirements.txt
```

Run everything from the repository root. W&B logging is on by default: run `wandb login`, or add `wandb.use_wandb=False`. For the NAS task, download `NAS-Bench-201-v1_0-e61699.pth` from [NAS-Bench-201](https://github.com/D-X-Y/NAS-Bench-201) into `notebooks/data/`.

## Training

Each run writes to `outputs/hydra/<task>/<date>/<time>/` (model in `model/`). One run takes about 70 hours on a V100. [slurm/train.sh](slurm/train.sh) is an example job script.

```bash
# base model, also the amortized expert that generates feedback for FICBO
python train_ficbo.py task=general_gp_prior file_name=gp_icbo_base.pth
BASE=outputs/hydra/general_gp_prior/<date>/<time>/model/gp_icbo_base.pth

# FICBO (Mixture / Additive / Combined prior)
python train_ficbo.py --config-name train_main task=ficbo_prior expert_path=$BASE file_name=ficbo_mixture.pth
python train_ficbo.py --config-name train_main task=additive_prior expert_bias=additive expert_path=$BASE file_name=ficbo_additive.pth
python train_ficbo.py --config-name train_main task=mixed_prior expert_path=$BASE file_name=ficbo_combined.pth

# baselines: GP-prior ICBO, GP-prior ICBO + feedback feature
python train_ficbo.py task=gp_icbo file_name=gp_icbo.pth
python train_ficbo.py task=gp_icbo_feedback_feature file_name=gp_icbo_feedback_feature.pth
```

FICBO (Ensemble) needs no training: it averages the Mixture and Additive policies at evaluation.
The defaults match the paper. Any other setting is a Hydra override, e.g. `task.k_decoy_max=8`.

## Evaluation

All paper experiments are in [notebooks/eval_with_expert.ipynb](notebooks/eval_with_expert.ipynb).
Set the model paths in its loading cells. [eval_benchmarks.py](eval_benchmarks.py) computes all synthetic-benchmark stats in one call.

## Code Structure

- `train_ficbo.py`: training
- `config/`: Hydra configs. `task/` holds the priors, `expert_bias/` the feedback distortions (there is many new ones, not mentioned in hte paper, but serve as additional examples one can use if your task/assumptions call for it)
- `model/`: transformer (embedder, encoder, policy and inference heads)
- `tasks/`: synthetic priors (`gaussian_process.py`) and benchmarks
- `expert_models/`: feedback sources and bias operators
- `utils/`: evaluation and baselines
- `notebooks/`: evaluation

## Citation

```bibtex
@misc{blumer2026incontextblackboxoptimizationunreliable,
      title={In-Context Black-Box Optimization with Unreliable Feedback},
      author={Nicolas Samuel Blumer and Julien Martinelli and Samuel Kaski},
      year={2026},
      eprint={2605.06187},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2605.06187},
}
```

Please also cite ALINE:

```bibtex
@inproceedings{huang2025aline,
  title={ALINE: Joint Amortization for Bayesian Inference and Active Data Acquisition},
  author={Huang, Daolang and Wen, Xinyi and Bharti, Ayush and Kaski, Samuel and Acerbi, Luigi},
  booktitle={Thirty-ninth Conference on Neural Information Processing Systems},
  year={2025}
}
```

## License

MIT, see [LICENSE](LICENSE). Includes code derived from ALINE (MIT).
