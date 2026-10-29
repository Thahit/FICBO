from abc import ABC, abstractmethod
import copy
import numpy as np
import torch
import torch.nn as nn
from joblib import Parallel, delayed
from expert_models.base_expert import Expert
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from types import SimpleNamespace as AttrDict


def build_sklearn_model(scfg):
    if scfg is None:
        raise ValueError("sklearn_model_cfg must be provided when expert_model_type='sklearn'")
    if scfg.type == "random_forest":
        return RandomForestRegressor(
            n_estimators = scfg.n_estimators,
            max_depth = scfg.max_depth,
            min_samples_split = scfg.min_samples_split,
            min_samples_leaf = scfg.min_samples_leaf,
            max_features = scfg.max_features,
            bootstrap = scfg.bootstrap,
            random_state = scfg.random_state,
            n_jobs = 1,
        )
    elif scfg.type == "xgboost":
        return XGBRegressor(
            n_estimators = scfg.n_estimators,
            max_depth = scfg.max_depth,
            learning_rate = scfg.learning_rate,
            subsample = scfg.subsample,
            colsample_bytree = scfg.colsample_bytree,
            min_child_weight = scfg.min_child_weight,
            gamma = scfg.gamma,
            reg_alpha = scfg.reg_alpha,
            reg_lambda = scfg.reg_lambda,
            random_state = scfg.random_state,
            verbosity = scfg.verbosity,
            n_jobs = 1,
        )
    else:
        raise ValueError(f"Unknown sklearn model type: {scfg.type}")


class ExpertModel(ABC):# was initially thinking of adding more
    """
    Interface for regression models used inside ModelExpert.
    Implement fit and predict to plug in any model.
    """

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit the model on training data.
        X : [N, d]
        y : [N]
        """
        ...

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict on new inputs.
        X       : [N, d]
        returns : [N]
        """
        ...

    @abstractmethod
    def clone(self) -> "ExpertModel":
        """Return an unfitted copy of this model."""
        ...


class SklearnExpertModel(ExpertModel):
    """
    Wraps any sklearn-compatible estimator (fit/predict API).
    Example:
        model = SklearnExpertModel(RandomForestRegressor(n_estimators=50))
    """

    def __init__(self, sklearn_model):
        self.model = sklearn_model

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def clone(self) -> "SklearnExpertModel":
        return SklearnExpertModel(copy.deepcopy(self.model))


class ModelExpert(Expert):
    """
    Expert that fits an ExpertModel independently per batch item,
    using y_hidden as the supervision signal.

    Parameters
    ----------
    model        : ExpertModel instance (unfitted) - cloned B times at init
    batch        : AttrDict from sample_batch
    fit_on       : which splits to train on - subset of
                   ["context", "query", "expert", "target"]
    as_probs     : if True, apply softmax over query dim before returning
    temperature  : softmax temperature (only used when as_probs=True)
    n_jobs       : joblib parallelism (-1 = all cores, 1 = sequential)
    adaptive     : if True, refit when add_data is called
    """

    def __init__(self, model, batch, fit_on=["expert"], as_probs=False,
             temperature=1.0, n_jobs=-1, adaptive=False, **kwargs):
        assert isinstance(model, ExpertModel)

        self.base_model = model
        self.fit_on = fit_on  # shared mutable default if caller omits fit_on; only ever read, never mutated in place - keep it that way
        self.as_probs = as_probs
        self.temperature = temperature
        self.n_jobs = n_jobs
        self.adaptive = adaptive
        self.expert_dim = batch.expert_dim_x
        self.batch_size = batch.context_x.shape[0]

        self._parallel = Parallel(n_jobs=n_jobs, prefer="processes")  # reused across all calls

        self.models = [model.clone() for _ in range(self.batch_size)]

        self._train_X = [None] * self.batch_size
        self._train_y = [None] * self.batch_size

        self._fit(batch)

    def _collect_fit_data(self, batch):
        xs, ys = [], []
        for split in self.fit_on:
            x_t = getattr(batch, f"{split}_x", None)
            y_t = getattr(batch, f"{split}_y", None)
            if x_t is None or x_t.shape[1] == 0:
                continue
            xs.append(x_t[..., -self.expert_dim:].cpu().numpy())
            ys.append(y_t.squeeze(-1).cpu().numpy())

        if not xs:
            return [None] * self.batch_size

        X_all = np.concatenate(xs, axis=1)  # [B, N_total, d]
        y_all = np.concatenate(ys, axis=1)  # [B, N_total]
        return [(X_all[b], y_all[b]) for b in range(self.batch_size)]


    @staticmethod
    def _fit_one(model: ExpertModel, X: np.ndarray, y: np.ndarray) -> ExpertModel:
        model.fit(X, y)
        return model

    def _fit(self, batch):
        data = self._collect_fit_data(batch)

        for b in range(self.batch_size):
            if data[b] is not None:
                self._train_X[b] = data[b][0]
                self._train_y[b] = data[b][1]

        valid = [(b, data[b]) for b in range(self.batch_size) if data[b] is not None]

        fitted = self._parallel(
            delayed(self._fit_one)(self.models[b], X, y)
            for b, (X, y) in valid
        )

        for (b, _), model in zip(valid, fitted):
            self.models[b] = model


    @staticmethod
    def _predict_one(model: ExpertModel, X: np.ndarray) -> np.ndarray:
        return model.predict(X)  # [N]

    def _predict_batch(self, x_tensor):
        x_np = x_tensor[..., -self.expert_dim:].cpu().numpy()  # [B, N, d]

        preds = self._parallel(
            delayed(self._predict_one)(self.models[b], x_np[b])
            for b in range(self.batch_size)
        )

        pred_tensor = torch.from_numpy(
            np.stack(preds, axis=0)
        ).to(dtype=x_tensor.dtype, device=x_tensor.device).unsqueeze(-1)  # [B, N, 1]

        if self.as_probs:
            scores = pred_tensor.squeeze(-1) / self.temperature
            pred_tensor = torch.softmax(scores, dim=-1).unsqueeze(-1)

        return pred_tensor

    def initial_predict(self, batch: AttrDict, augment_queries: bool = True, **kwargs):
        """
        Predict feedback for context points.
        If augment_queries=True, also runs on query points internally
        (matching AmortizedExpert behaviour) but returns context slice only.
        Returns [B, n_context, 1].
        """
        if augment_queries:
            x_all = torch.cat([batch.context_x, batch.query_x], dim=1)
            out = self._predict_batch(x_all)
            out = out[:, :batch.context_x.shape[1]]
        else:
            out = self._predict_batch(batch.context_x)

        return out  # [B, n_context, 1]

    def predict(self, x: torch.Tensor, time_token=None, **kwargs) -> torch.Tensor:
        """
        Predict for arbitrary inputs x [B, N, dim].
        Returns [B, N, 1].
        """
        return self._predict_batch(x)

    def add_data(self, x_new, y_new, **kwargs):
        if not self.adaptive:
            return

        x_np = x_new[..., -self.expert_dim:].cpu().numpy()  # [B, 1, d]
        y_np = y_new.squeeze(-1).cpu().numpy()  # [B, 1]

        for b in range(self.batch_size):
            if self._train_X[b] is not None:
                self._train_X[b] = np.concatenate([self._train_X[b], x_np[b]], axis=0)
                self._train_y[b] = np.concatenate([self._train_y[b], y_np[b]], axis=0)
            else:
                self._train_X[b] = x_np[b]
                self._train_y[b] = y_np[b]

        valid = [(b, self._train_X[b], self._train_y[b])
                for b in range(self.batch_size)
                if self._train_X[b] is not None]

        fitted = self._parallel(
            delayed(self._fit_one)(self.models[b], X, y)  # no clone
            for b, X, y in valid
        )

        for (b, _, _), model in zip(valid, fitted):
            self.models[b] = model

    def __str__(self):
        return (f"ModelExpert(model={type(self.base_model).__name__}, "
                f"fit_on={self.fit_on}, as_probs={self.as_probs}, "
                f"expert_dim={self.expert_dim}, batch_size={self.batch_size}, "
                f"adaptive={self.adaptive})")
