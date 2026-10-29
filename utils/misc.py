import torch
from torch import optim
from torch.optim import lr_scheduler
import os
import numpy as np
import random
from omegaconf import OmegaConf
import hydra
from hydra import initialize_config_dir, compose
from model.base import FICBO

def set_seed(seed):
    """Seeds torch/cuda/numpy/random and forces cudnn into deterministic mode."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_state_dict(model, dir, name="ficbo.pth"):
    """Saves to <dir>/model/<name>, creating the model/ subdir if needed."""
    file_path = os.path.join(dir, 'model')
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    file_path = os.path.join(file_path, name)
    torch.save(model.state_dict(), file_path)
    return file_path


def load_state_dict(model, dir, name="ficbo.pth"):
    """Counterpart to save_state_dict - loads from <dir>/model/<name>."""
    file_path = os.path.join(dir, 'model', name)
    model.load_state_dict(torch.load(file_path, map_location=torch.get_default_device(), weights_only=True))
    return model


def save_checkpoint(cfg, model, optimizer, scheduler, epoch, with_epoch=False):
    """Also stashes RNG state so a resumed run continues the same data stream.

    with_epoch=True suffixes the filename, e.g. ckptname_1000.tar.
    """
    ckpt = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch": epoch,
        'rng_state': torch.get_rng_state(),
        'cuda_rng_state': torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
        'numpy_rng_state': np.random.get_state(),
        'random_rng_state': random.getstate(),
    }

    if with_epoch:
        checkpoint_name = f"{cfg.checkpoint_name.split('.')[0]}_{epoch}.tar"
    else:
        checkpoint_name = cfg.checkpoint_name

    checkpoint_path = os.path.join(cfg.output_dir, checkpoint_name)
    torch.save(ckpt, str(checkpoint_path))


def load_checkpoint(cfg, model, optimizer, scheduler, ckpt_path=None, check_layerwise=True):
    """Restores model/optimizer/scheduler plus RNG state; rebuilds the optimizer via
    set_layerwise_lr first if check_layerwise is set, since its param groups depend on epoch.
    """
    if ckpt_path is None:
        ckpt_path = os.path.join(cfg.output_dir, cfg.checkpoint_name)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint file not found: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=torch.get_default_device(), weights_only=False)
    try:
        model.load_state_dict(ckpt["model"])
    except KeyError:
        model.load_state_dict(ckpt)
    epoch = ckpt["epoch"]

    if check_layerwise:
        optimizer, scheduler = set_layerwise_lr(cfg, model, epoch - 1)  # param groups of last epoch

    optimizer.load_state_dict(ckpt["optimizer"])
    scheduler.load_state_dict(ckpt["scheduler"])
    if not isinstance(ckpt['rng_state'], torch.ByteTensor):
        ckpt['rng_state'] = torch.ByteTensor(ckpt['rng_state'].cpu())
    torch.set_rng_state(ckpt['rng_state'])

    if torch.cuda.is_available() and ckpt['cuda_rng_state'] is not None:
        if not isinstance(ckpt['cuda_rng_state'], torch.ByteTensor):
            ckpt['cuda_rng_state'] = torch.ByteTensor(ckpt['cuda_rng_state'].cpu())
        torch.cuda.set_rng_state(ckpt['cuda_rng_state'])

    np.random.set_state(ckpt['numpy_rng_state'])
    random.setstate(ckpt['random_rng_state'])

    return epoch, optimizer, scheduler


def set_layerwise_lr(cfg, model, epoch=0):
    """Before cfg.burning_epoch, one lr for everything. After, non-predictor ("shared")
    params get lr/5 while predictor params keep the full lr.
    """
    if epoch < cfg.burning_epoch:
        optimizer = getattr(optim, cfg.optimizer)(
            model.parameters(),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
        )
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg.max_epoch
        )    
    else:
        shared_params, predictor_params = [], []
        for name, param in model.named_parameters():
            if 'predictor' not in name:
                shared_params.append(param)
            else:
                predictor_params.append(param)

        optimizer = getattr(optim, cfg.optimizer)(
            [
                {"params": shared_params, "lr": cfg.lr / 5},
                {"params": predictor_params},
            ],
            lr=cfg.lr,
        )
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg.max_epoch - cfg.burning_epoch
        )
    return optimizer, scheduler


def load_config_and_model(path, config_name="config.yaml", file_name="ficbo.pth", 
                          load_type="ckpt", device=None):
    """Rebuilds a FICBO model from a run's hydra config dir and loads its weights.
    path can be relative to the project root or absolute. load_type "ckpt" expects a
    dict with a "model" key (falls back to a raw state dict); "pth" is a plain state dict.
    """

    if os.path.isabs(path):
        full_dir = path
    else:
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
        full_dir = os.path.abspath(os.path.join(project_root, path))

    config_dir = os.path.join(full_dir, ".hydra")

    if not os.path.isdir(config_dir):
        raise FileNotFoundError(f"Config path not found: {config_dir}")

    # abs path, hydra resolves relative to module
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(config_name=config_name)
        resolved_cfg = OmegaConf.create(OmegaConf.to_yaml(cfg, resolve=True))

        if device is not None:
            torch.set_default_device(device)

        embedder = hydra.utils.instantiate(cfg.embedder)
        encoder = hydra.utils.instantiate(cfg.encoder)
        head = hydra.utils.instantiate(cfg.head)
        model = FICBO(embedder, encoder, head)

        file_path = os.path.join(full_dir, file_name)
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Model file not found: {file_path}")
        
        try:
            if load_type.lower() == "ckpt":
                ckpt = torch.load(file_path, map_location=torch.get_default_device(), weights_only=False)
                
                if isinstance(ckpt, dict) and "model" in ckpt:
                    model.load_state_dict(ckpt["model"])
                else:
                    model.load_state_dict(ckpt)
                    
            elif load_type.lower() == "pth":
                state_dict = torch.load(file_path, map_location=torch.get_default_device(), weights_only=True)
                model.load_state_dict(state_dict)
                
            else:
                raise ValueError(f"Invalid load_type: {load_type}. Must be 'ckpt' or 'pth'")
                
        except Exception as e:
            raise RuntimeError(f"Failed to load model from {file_path}: {str(e)}")
    
    return resolved_cfg, model


def calculate_gmm_variance(mixture_means, mixture_stds, mixture_weights):
    """
    Total variance of a per-query Gaussian mixture: [B,Q,K] means/stds/weights -> [B,Q].
    weights can also come in as [B,K] (shared across queries) and gets broadcast.
    """
    B, Q, K = mixture_means.shape

    if len(mixture_weights.shape) == 2:  # [B,K] -> [B,Q,K]
        weights = mixture_weights.unsqueeze(1).expand(B, Q, K)
    else:
        weights = mixture_weights

    weighted_means = torch.sum(weights * mixture_means, dim=2)  # [B,Q]

    # var = sum_k w_k * (std_k^2 + (mean_k - weighted_mean)^2)
    mean_diffs_squared = (mixture_means - weighted_means.unsqueeze(2)) ** 2  # [B,Q,K]
    var = torch.sum(weights * (mixture_stds ** 2 + mean_diffs_squared), dim=2)  # [B,Q]

    return var
