import torch
from torch import optim
from torch.optim import lr_scheduler
from torch.nn.utils import clip_grad_norm_
import os
import random
import numpy as np
import time
import wandb
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf

from model import FICBO
from utils import (create_logger, set_seed, save_state_dict, 
                   compute_ll, load_checkpoint, save_checkpoint, 
                   create_target_mask, select_targets_by_mask, 
                   set_layerwise_lr, eval_boed
                    )


def summarize_problem(batch, threshold_ratio=0.95):
    """
    Summarize the current batch/problem.

    Args:
        batch.query_y: [B, N, 1] tensor of possible action values
        threshold_ratio: fraction of max to consider "close to max"

    Returns: dict with useful statistics
        - avg_max: average max value over batch
        - avg_val: average over all points
        - n_close_to_max: average number of points per batch that are >= threshold_ratio * max
        - std_val: std of values
        - max_indices: argmax indices per batch
        - max_gap: difference between best and mean of other points
    """
    query_y = batch.query_y.squeeze(-1)  # [B, N]

    max_vals, max_idx = query_y.max(dim=1)  # [B]
    avg_max = max_vals.mean().item()
    y_min = query_y.min(dim=1).values.unsqueeze(1)

    avg_val = query_y.mean(dim=1).mean().item()
    std_val = query_y.std(dim=1).mean().item()

    threshold = y_min + threshold_ratio * (max_vals.unsqueeze(1) - y_min)
    n_close = (query_y >= threshold).sum(dim=1).float().mean().item()

    # Gap between max and mean of others
    mean_others = (query_y.sum(dim=1) - max_vals) / (query_y.size(1) - 1)
    max_gap = (max_vals - mean_others).mean().item()

    return {
        "avg_max": round(avg_max, 4),
        "avg_val": round(avg_val, 4),
        "std_val": round(std_val, 4),
        "n_close_to_max": round(n_close, 2),
        "gap_max_to_avg": round(max_gap, 4),
        "avg_min": round(y_min.mean().item(), 4),
    }


def discounted_cumsum(x, gamma):
    """Discounted cumulative sum along dim 1 (time), same shape in and out: [B, T]."""
    T = x.size(1)
    powers = gamma ** torch.arange(T, device=x.device)
    return torch.flip(
        torch.cumsum(torch.flip(x * powers, dims=[1]), dim=1),
        dims=[1]
    ) / powers


def train(cfg, logger, model, experiment, batch_size: int, min_T: int, max_T: int, max_epoch: int, 
        verbose: int = 10, loss_on_all_target: bool = False, pred_loss_on_query: bool = False):
    """ Multitask Learning with flexible reward setup for active learning / BO.

        Computes three types of rewards:
            - nll-based (posterior improvement)
            - max-value (final best)
            - stepwise improvement (incremental BO-style)
        Combines them via config weights: alpha_nll, alpha_max, alpha_impr.

        Args:
            batch_size (int): number of parallel experiments
            min_T (int): minimum value of T
            max_T (int): maximum value of T
            max_epoch (int): max epoch of optimisation
            verbose (int): interval to show the process.
    """
    optimizer, scheduler = set_layerwise_lr(cfg, model)

    logger.info(f"Optimizer: {optimizer}")
    logger.info(f"Scheduler: {scheduler}")

    if cfg.load_checkpoint is True:
        start_epoch, optimizer, scheduler = load_checkpoint(cfg, model, optimizer, scheduler, cfg.load_path)
    else:
        start_epoch = 0

    logger.info(f"Optimizer: {optimizer}")
    logger.info(f"Scheduler: {scheduler}")

    if start_epoch < cfg.burning_epoch:
        experiment.n_query_init = cfg.T

    losses = []
    training_times = []

    for epoch in range(start_epoch, max_epoch):
        start_time = time.time()
        model.train()
        optimizer.zero_grad()

        T = random.randint(min_T, max_T)
        batch = experiment.sample_batch(batch_size)

        mask_type = random.choice(cfg.task.mask_type)
        batch.target_mask = create_target_mask(
            mask_type,
            cfg.task.embedding_type,
            cfg.task.n_target_data,
            cfg.task.n_target_theta,
            cfg.task.n_selected_targets,
            cfg.task.predefined_masks,
            cfg.task.predefined_mask_weights,
            cfg.task.mask_index,
            cfg.task.attend_to,
        )

        device = batch.target_all.device

        # Initialize logs (PREALLOCATED)
        log_probs = torch.zeros((batch_size, T), device=device)
        nlls_for_prediction = torch.zeros((batch_size, T), device=device)
        nlls_for_query = torch.zeros((batch_size, T), device=device)

        agreements_hard = []
        agreements_soft = []
        mean_y_diffs = []
        mean_y_diffs_sampled = []
        expert_best_hits = []
        expert_rank = []

        # Reward trackers (PREALLOCATED)
        R_nll = torch.zeros((batch_size, T), device=device)
        R_impr = torch.zeros((batch_size, T), device=device)

        best_so_far = batch.context_y[:, 0]
        true_best = torch.max(batch.query_y.max(dim=1).values.squeeze(-1), best_so_far.squeeze(-1))

        if epoch % verbose == 0:
            print(summarize_problem(batch))

        for t in range(T):
            if cfg.time_token:
                batch.t = torch.tensor([t / T], device=device)

            #print(f"t={t} mem: {torch.cuda.memory_allocated()/1e9:.2f}GB")
            #torch.cuda.memory._record_memory_history(max_entries=100000)
            pred = model.forward(batch)
            #print(f"t={t} after forward: {torch.cuda.memory_allocated()/1e9:.2f}GB")
            #torch.cuda.memory._dump_snapshot("mem_snapshot.pkl")
            #torch.cuda.memory._record_memory_history(enabled=None) #use this after: python -m torch.cuda._memory_viz trace_plot mem_snapshot.pkl -o mem_snapshot.html

            design_out = pred.design_out
            posterior_out = pred.posterior_out
            posterior_of_queries = pred.posterior_out_query

            y_t = batch.query_y[torch.arange(batch_size), design_out.idx.squeeze(-1)]

            if getattr(cfg.task, "use_expert_model", False):
                with torch.no_grad():
                    expert_probs = batch.query_feedback[..., 0]
                    if getattr(cfg.task, "normalize_outputs_to_max_one", False):
                        expert_probs = expert_probs / expert_probs.sum(dim=-1, keepdim=True)

                    policy_probs = design_out.zt
                    policy_actions = policy_probs.argmax(dim=-1)
                    expert_actions = expert_probs.argmax(dim=-1)

                    agreements_hard.append((policy_actions == expert_actions).float().mean())
                    agreements_soft.append((policy_probs * expert_probs).sum(dim=-1).mean())

                    qy_full = batch.query_y
                    model_y = qy_full[torch.arange(qy_full.size(0)), policy_actions].squeeze(-1)
                    expert_y = qy_full[torch.arange(qy_full.size(0)), expert_actions].squeeze(-1)

                    mean_y_diffs.append((model_y - expert_y).mean())
                    mean_y_diffs_sampled.append((y_t - expert_y).mean())

                    qy = batch.query_y.squeeze(-1)
                    true_best_idx = qy.argmax(dim=1)

                    expert_best_hits.append((expert_actions == true_best_idx).float().mean())
                    rank = (qy > expert_y.unsqueeze(-1)).sum(dim=1) + 1
                    expert_rank.append(rank.float().mean())

            log_probs[:, t] = design_out.log_prob

            target_ll = compute_ll(
                batch.target_all,
                posterior_out.mixture_means,
                posterior_out.mixture_stds,
                posterior_out.mixture_weights,
            )
            if loss_on_all_target:
                masked_target_ll = target_ll
            else:
                masked_target_ll = select_targets_by_mask(target_ll, batch.target_mask)
            

            if cfg.task.embedding_type == "mix" and mask_type == "all":
                nll_for_query = -(
                    masked_target_ll[:, :-cfg.task.n_target_theta].mean(dim=-1)
                    + masked_target_ll[:, -cfg.task.n_target_theta:].mean(dim=-1)
                )
            else:
                nll_for_query = -masked_target_ll.mean(dim=-1)
            
            if pred_loss_on_query:
                query_ll = compute_ll(
                    batch.query_y,
                    posterior_of_queries.mixture_means,
                    posterior_of_queries.mixture_stds,
                    posterior_of_queries.mixture_weights,
                )
                nll_for_query = nll_for_query*.5 + query_ll.mean(dim=-1)*.5

            nlls_for_query[:, t] = nll_for_query # guess the variable names are questionable

            if cfg.task.embedding_type == "mix":
                nll = -(
                    target_ll[:, :-cfg.task.n_target_theta].mean(dim=-1)
                    + target_ll[:, -cfg.task.n_target_theta:].mean(dim=-1)
                )
            else:
                nll = -target_ll.mean(dim=-1)

            nlls_for_prediction[:, t] = nll
            
            batch = experiment.update_batch(batch, design_out.idx)

            impr = torch.clamp(y_t - best_so_far, min=0.0).detach()
            R_impr[:, t] = impr.squeeze(-1)

            best_so_far = torch.max(best_so_far, y_t)

            if t > 0:
                ll_gain = torch.clamp(nlls_for_query[:, t - 1] - nll_for_query, min=0.0).detach()
            else:
                ll_gain = torch.zeros_like(nll_for_query)

            R_nll[:, t] = ll_gain
            del pred, design_out, posterior_out, nll_for_query, nll, target_ll
            #print(f"t={t} after del: {torch.cuda.memory_allocated()/1e9:.2f}GB")


        R_nll = discounted_cumsum(R_nll, cfg.gamma)
        R_impr = discounted_cumsum(R_impr, cfg.gamma)

        gamma_powers = cfg.gamma ** torch.arange(T - 1, -1, -1, device=device)
        R_max = best_so_far.detach() * gamma_powers.unsqueeze(0)

        R_nll_norm = (R_nll - R_nll.mean(dim=0, keepdim=True)) / (R_nll.std(dim=0, keepdim=True) + 1e-8)
        R_impr_norm = (R_impr - R_impr.mean(dim=0, keepdim=True)) / (R_impr.std(dim=0, keepdim=True) + 1e-8)
        R_max_norm = best_so_far.detach() / true_best.unsqueeze(-1)

        alpha_nll = getattr(cfg, "alpha_nll", 1.0)
        alpha_max = getattr(cfg, "alpha_max", 0)
        alpha_impr = getattr(cfg, "alpha_impr", 0)

        R_combined = torch.zeros_like(R_nll_norm)

        if alpha_nll > 0:
            R_combined += alpha_nll * R_nll_norm
        if alpha_max > 0:
            R_combined += alpha_max * R_max_norm
        if alpha_impr > 0:
            R_combined += alpha_impr * R_impr_norm

        log_probs = torch.clamp(log_probs, min=-10.0)
        design_loss = -torch.mean(log_probs * R_combined)

        predict_loss = nlls_for_prediction.mean()

        if epoch < cfg.burning_epoch:
            loss = predict_loss
            loss.backward()
        else:
            loss = design_loss * cfg.alpha + predict_loss
            loss.backward()

        losses.append(loss.item())

        if cfg.clip_grads:
            clip_grad_norm_(model.parameters(), max_norm=1.0)

        if epoch == cfg.burning_epoch:
            optimizer, scheduler = set_layerwise_lr(cfg, model, epoch)
            experiment.n_query_init = cfg.task.n_query_init
            logger.info(
                f"""Model has been saved at {save_state_dict(model, cfg.output_dir, f'{cfg.file_name.split(".")[0]}_burning.pth')}"""
            )

        optimizer.step()
        scheduler.step()

        training_times.append(time.time() - start_time)

        best_so_far = best_so_far.squeeze(-1)

        if cfg.wandb.use_wandb and epoch % cfg.wandb.log_freq == 0:
            wandb.log(
                {
                    "loss": loss,
                    "likelihood": -predict_loss,
                    "design_loss": design_loss,
                    "targeted_likelihood": -nlls_for_query.mean(),
                    "reward_nll_mean": R_nll.mean(),
                    "reward_max_mean": R_max.mean(),
                    "reward_impr_mean": R_impr.mean(),
                    "best_so_far": best_so_far.mean(),
                    "mean_regret": (torch.clamp(true_best - best_so_far, min=0)).float().mean(),
                    "found_optimal": (best_so_far >= true_best * 0.95).float().mean(),
                    "agreement with expert hard": torch.stack(agreements_hard).mean()
                    if getattr(cfg.task, "use_expert_model", False)
                    else 0,
                    "agreements  with expert soft": torch.stack(agreements_soft).mean()
                    if getattr(cfg.task, "use_expert_model", False)
                    else 0,
                    "mean_y_diffs of policy argmax to expert argmax": torch.stack(mean_y_diffs).mean()
                    if getattr(cfg.task, "use_expert_model", False)
                    else 0,
                    "mean_y_diffs  of policy sample to expert argmax": torch.stack(mean_y_diffs_sampled).mean()
                    if getattr(cfg.task, "use_expert_model", False)
                    else 0,
                    "expert_best_hit_rate": torch.stack(expert_best_hits).mean()
                    if getattr(cfg.task, "use_expert_model", False)
                    else 0,
                    "expert_avg_rank": torch.stack(expert_rank).mean()
                    if getattr(cfg.task, "use_expert_model", False)
                    else 0,
                    "active_prior_idx": getattr(experiment, "active_prior_idx", 0),
                },
                step=epoch,
            )

        if epoch % verbose == 0:
            logger.info(
                f"Epoch: {epoch}, loss: {losses[-1]:.4f}, T: {T}, "
                f"likelihood: {-predict_loss}, design_loss: {design_loss}, predict_loss: {predict_loss}"
            )
            if cfg.eval.EIG:
                bounds = eval_boed(model, experiment, cfg.T - cfg.task.n_context_init, cfg.eval.L, cfg.eval.M, cfg.eval.batch_size, cfg.time_token, False)
                pce_loss = bounds['pce_mean']
                nmc_loss = bounds['nmc_mean']
                logger.info(f"PCE: {pce_loss}\tNMC: {nmc_loss}")
                if cfg.wandb.use_wandb:
                    wandb.log({"PCE": pce_loss, "NMC": nmc_loss}, step=epoch)
                    
        next_epoch = epoch + 1
        if cfg.checkpoint and next_epoch % cfg.checkpoint == 0:
            save_checkpoint(cfg, model, optimizer, scheduler, next_epoch, with_epoch=True)
        del batch, loss
        torch.cuda.empty_cache()



    total_time = sum(training_times)
    burn_in_offset = max(0, cfg.burning_epoch - start_epoch)  # training_times[i] is real epoch start_epoch + i
    average_time = np.mean(training_times[burn_in_offset:])
    std_time = np.std(training_times[burn_in_offset:])
    logger.info(
        f"Total training time: {total_time:.2f} seconds ({total_time / 3600:.2f} hours), "
        f"average time per epoch: {average_time:.2f}+-{std_time:.2f} seconds"
    )
    
    if cfg.wandb.use_wandb:
        wandb.log({"training_time": total_time}, step=max_epoch)

    return losses


@hydra.main(version_base=None, config_path="./config", config_name="train")
def main(cfg):
    if not torch.cuda.is_available():
        cfg.device = "cpu"
    torch.set_default_device(cfg.device)
    if cfg.device == "cuda":
        torch.set_default_tensor_type('torch.cuda.FloatTensor')

    if cfg.fix_seed:    # Setting random seed
        set_seed(cfg.seed)
    else:
        cfg.seed = torch.random.seed()

    cfg.output_dir = str(HydraConfig.get().runtime.output_dir)

    # Ensure min_T is not larger than T
    if cfg.min_T > cfg.T:
        cfg.min_T = cfg.T
    
    # Create logger
    logger = create_logger(os.path.join(cfg.output_dir, 'logs'), name=cfg.task.name)
    logger.info("Running with config:\n{}".format(OmegaConf.to_yaml(cfg)))

    if cfg.wandb.use_wandb:
        run = wandb.init(
            project=cfg.wandb.project,
            name=cfg.wandb.run_name,
            group=cfg.wandb.group,
            config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),
            dir=cfg.output_dir,
            tags=cfg.wandb.get("tags", None),
        )
        # Save hydra configs with wandb (handles hydra's multirun dir)
        try:
            hydra_log_dir = os.path.join(HydraConfig.get().runtime.output_dir, ".hydra")
            wandb.save(str(hydra_log_dir), policy="now")
        except FileExistsError:
            pass

    # Data
    expert_model = None
    if getattr(cfg.task, "use_expert_model", False):
        if getattr(cfg, "expert_bias", None) is not None:
            expert_bias_model = hydra.utils.instantiate(cfg.expert_bias, device=cfg.device)
        else:
            expert_bias_model = None

        print("Loading expert model...")
        if getattr(cfg.task, "expert_model_type", "truth") != "amortized":
            expert_model=None # no expert needed
            print("no expert model loaded, using ground truth")
        else:
            embedder = hydra.utils.instantiate(cfg.embedder_for_expert)
            encoder = hydra.utils.instantiate(cfg.encoder_for_expert)
            head = hydra.utils.instantiate(cfg.head_for_expert)
            expert_model = FICBO(embedder, encoder, head)
            ckpt = torch.load(cfg.expert_path, map_location=torch.get_default_device(), weights_only=False)
            try:
                expert_model.load_state_dict(ckpt["model"])
            except KeyError:
                expert_model.load_state_dict(ckpt)
            for p in expert_model.parameters():
                p.requires_grad = False
            expert_model.eval()
            expert_model.embedding_type = "data"

        #load weights
        experiment = hydra.utils.instantiate(cfg.task, expert_model=expert_model,
                                             expert_bias_model=expert_bias_model, model_expert_cfg=cfg.get("expert_model", None))
    else:
        experiment = hydra.utils.instantiate(cfg.task)
    logger.info(f"Task: {experiment}")

    # For HPO tasks, validate that we have an HPO task and update config dimensions
    if hasattr(experiment, 'meta_dataset'):
        logger.info(f"Using HPO-B meta-dataset: {experiment.meta_dataset}")
        logger.info(f"Input dimension: {experiment.dim_x}")
        logger.info(f"Number of datasets: {experiment.hpob.n_dataset}")

        # Update config with actual dimensions from the dataset
        if cfg.task.dim_x != experiment.dim_x:
            logger.info(
                f"Updating dim_x from config value {cfg.task.dim_x} to actual dataset dimension {experiment.dim_x}")
            cfg.task.dim_x = experiment.dim_x

        if cfg.task.dim_y != experiment.dim_y:
            logger.info(
                f"Updating dim_y from config value {cfg.task.dim_y} to actual dataset dimension {experiment.dim_y}")
            cfg.task.dim_y = experiment.dim_y
    
    
    # Model
    embedder = hydra.utils.instantiate(cfg.embedder)
    encoder = hydra.utils.instantiate(cfg.encoder)
    head = hydra.utils.instantiate(cfg.head)
    model = FICBO(embedder, encoder, head)
    logger.info(model)

    if cfg.wandb.use_wandb:
        wandb.watch(model, log_freq=200)

    # Train
    train(cfg, logger, model, experiment, cfg.batch_size, cfg.min_T, cfg.T, cfg.max_epoch, verbose=cfg.verbose, 
            loss_on_all_target= getattr(cfg, "loss_on_all_target", False),
            pred_loss_on_query= getattr(cfg, "pred_loss_on_query", False),
            )

    # Save
    logger.info(f"Model has been saved at {save_state_dict(model, cfg.output_dir, cfg.file_name)}")

    # Eval
    if cfg.eval.EIG: # that is ALINE leftover, has not been tested
        # Set a larger query size
        experiment.n_query_init = cfg.eval.n_query_final

        bounds = eval_boed(model, experiment, cfg.eval.T_final - cfg.task.n_context_init, cfg.eval.L_final, cfg.eval.M_final, cfg.eval.batch_size_final, cfg.time_token, stepwise=True)

        logger.info(bounds)
        logger.info(f"PCE: {bounds['pce_mean'][cfg.T-1]:.3f}+-{bounds['pce_err'][cfg.T-1]:.3f}\tNMC: {bounds['nmc_mean'][cfg.T-1]:.3f}+-{bounds['nmc_err'][cfg.T-1]:.3f}")

        # save bounds to file
        save_path = os.path.join(cfg.output_dir, "eval", f"{cfg.file_name.split('.')[0]}_N{cfg.eval.n_query_final}_T{cfg.eval.T_final}.tar")
        # make dir if not exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(bounds, save_path)
        logger.info(f"Bounds have been saved at {save_path}.")


if __name__ == '__main__':
    main()
