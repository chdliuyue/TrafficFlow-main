
"""
myloss.py

Def-based loss functions to match your project's style (similar to mae.py).
- Mask semantics: if targets_mask is provided (0/1 or bool), we normalize it by mean(mask)
  so the expected scale of loss does not depend on the number of valid entries.
- Supports:
    * point losses: MAE / MSE / Huber
    * probabilistic losses: Gaussian/Laplace/StudentT NLL; Quantile pinball; (optional domain-restricted families)
    * regularization: graph basis orthogonality/sparsity, smoothness of graph scales, fusion weight sparsity

Expected model forward_return (outputs dict) keys when using distribution loss:
  outputs["prediction"]: [B,O,N]  (point prediction used by point loss)
  outputs["dist_name"]:   str
  outputs["dist_params"]: dict
Optional keys for regularization:
  outputs["graph_basis"]:  [N,r]
  outputs["graph_scales"]: [B,O,r]
  outputs["fusion_weights"]: dict of scalar tensors (e.g. {"w_base":..., "w_graph":...})
"""

import math
from dataclasses import dataclass
from typing import Dict, Optional

import torch


# =========================
# Mask utilities (aligned with mae.py)
# =========================

def normalize_mask(targets: torch.Tensor, targets_mask: Optional[torch.Tensor], eps: float = 1e-6) -> torch.Tensor:
    """
    Return a float mask with mean(mask)=1 over valid entries.
    - If targets_mask is None: all ones.
    - If bool mask: converted to float.
    """
    mask = targets_mask if targets_mask is not None else torch.ones_like(targets)
    mask = mask.float()
    mask = mask / torch.mean(mask).clamp_min(eps)
    mask = torch.nan_to_num(mask)
    return mask


def masked_reduce(loss: torch.Tensor, targets: torch.Tensor, targets_mask: Optional[torch.Tensor], eps: float = 1e-6) -> torch.Tensor:
    m = normalize_mask(targets, targets_mask, eps=eps)
    loss = torch.nan_to_num(loss) * m
    loss = torch.nan_to_num(loss)
    return torch.mean(loss)


# =========================
# Point losses
# =========================

def masked_mae(prediction: torch.Tensor, targets: torch.Tensor, targets_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    return masked_reduce(torch.abs(prediction - targets), targets, targets_mask)


def masked_mse(prediction: torch.Tensor, targets: torch.Tensor, targets_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    return masked_reduce((prediction - targets) ** 2, targets, targets_mask)


def masked_huber(
    prediction: torch.Tensor,
    targets: torch.Tensor,
    delta: float = 1.0,
    targets_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    d = prediction.new_tensor(float(delta))
    err = prediction - targets
    abs_err = torch.abs(err)
    quad = torch.minimum(abs_err, d)
    lin = abs_err - quad
    loss = 0.5 * quad ** 2 + d * lin
    return masked_reduce(loss, targets, targets_mask)


def compute_point_loss(
    prediction: torch.Tensor,
    targets: torch.Tensor,
    targets_mask: Optional[torch.Tensor],
    point_loss: str = "mae",
    huber_delta: float = 1.0,
) -> torch.Tensor:
    name = str(point_loss).lower()
    if name == "mae":
        return masked_mae(prediction, targets, targets_mask)
    if name == "mse":
        return masked_mse(prediction, targets, targets_mask)
    if name == "huber":
        return masked_huber(prediction, targets, delta=huber_delta, targets_mask=targets_mask)
    raise ValueError(f"Unknown point_loss: {point_loss} (expected mae|mse|huber)")


# =========================
# Likelihood / distribution losses
# =========================

def gaussian_nll(targets: torch.Tensor, mu: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return 0.5 * math.log(2.0 * math.pi) + torch.log(scale) + 0.5 * ((targets - mu) / scale) ** 2


def laplace_nll(targets: torch.Tensor, mu: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return math.log(2.0) + torch.log(scale) + torch.abs(targets - mu) / scale


def studentt_nll(targets: torch.Tensor, df: torch.Tensor, mu: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return -torch.distributions.StudentT(df=df, loc=mu, scale=scale).log_prob(targets)


def lognormal_nll(targets: torch.Tensor, mu: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return -torch.distributions.LogNormal(loc=mu, scale=scale).log_prob(targets)


def gamma_nll(targets: torch.Tensor, mu_pos: torch.Tensor, shape: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    # Gamma(concentration=shape, rate=shape/mu) so mean=mu
    rate = shape / (mu_pos + eps)
    return -torch.distributions.Gamma(concentration=shape, rate=rate).log_prob(targets)


def negbinom_nll(targets: torch.Tensor, mu_pos: torch.Tensor, total_count: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    # NegativeBinomial(total_count=r, probs=r/(r+mu)) so mean=mu
    probs = total_count / (total_count + mu_pos + eps)
    return -torch.distributions.NegativeBinomial(total_count=total_count, probs=probs).log_prob(targets)


def quantile_pinball(targets: torch.Tensor, quantiles: torch.Tensor, q_levels: torch.Tensor) -> torch.Tensor:
    """
    targets:   [B,O,N]
    quantiles: [B,O,N,Q]
    q_levels:  [Q]
    """
    e = targets.unsqueeze(-1) - quantiles
    ql = q_levels.view(1, 1, 1, -1).to(e.device).to(e.dtype)
    pin = torch.maximum(ql * e, (ql - 1.0) * e)  # [B,O,N,Q]
    return pin.mean(dim=-1)  # [B,O,N]


def compute_distribution_loss(
    targets: torch.Tensor,
    targets_mask: Optional[torch.Tensor],
    dist_name: str,
    dist_params: Dict[str, torch.Tensor],
    eps: float = 1e-6,
    check_domain: bool = True,
) -> torch.Tensor:
    dist = str(dist_name).lower()

    if dist == "none":
        return targets.new_tensor(0.0)

    if dist == "gaussian":
        nll = gaussian_nll(targets, mu=dist_params["mu"], scale=dist_params["scale"])
        return masked_reduce(nll, targets, targets_mask, eps=eps)

    if dist == "laplace":
        nll = laplace_nll(targets, mu=dist_params["mu"], scale=dist_params["scale"])
        return masked_reduce(nll, targets, targets_mask, eps=eps)

    if dist == "studentt":
        nll = studentt_nll(targets, df=dist_params["df"], mu=dist_params["mu"], scale=dist_params["scale"])
        return masked_reduce(nll, targets, targets_mask, eps=eps)

    if dist == "quantile":
        qloss = quantile_pinball(targets, dist_params["quantiles"], dist_params["q_levels"])
        if "cross_penalty" in dist_params:
            qloss = qloss + dist_params["cross_penalty"]
        return masked_reduce(qloss, targets, targets_mask, eps=eps)

    # domain-restricted families (only valid if targets are in the original positive domain)
    if dist == "lognormal":
        if check_domain and (targets <= 0).any():
            raise ValueError("LogNormal likelihood requires positive targets in original domain.")
        nll = lognormal_nll(targets, mu=dist_params["mu"], scale=dist_params["scale"])
        return masked_reduce(nll, targets, targets_mask, eps=eps)

    if dist == "gamma":
        if check_domain and (targets <= 0).any():
            raise ValueError("Gamma likelihood requires positive targets in original domain.")
        nll = gamma_nll(targets, mu_pos=dist_params["mu_pos"], shape=dist_params["shape"], eps=eps)
        return masked_reduce(nll, targets, targets_mask, eps=eps)

    if dist == "negbinom":
        if check_domain and (targets < 0).any():
            raise ValueError("NegativeBinomial likelihood requires nonnegative targets in original domain.")
        nll = negbinom_nll(targets, mu_pos=dist_params["mu_pos"], total_count=dist_params["total_count"], eps=eps)
        return masked_reduce(nll, targets, targets_mask, eps=eps)

    raise ValueError(f"Unsupported likelihood: {dist}")


# =========================
# Regularization
# =========================

@dataclass
class RegWeights:
    reg_graph_orth: float = 0.0
    reg_graph_l1: float = 0.0
    reg_graph_scale_smooth: float = 0.0
    reg_fusion_l1: float = 0.0



def compute_regularization(outputs: Dict, reg: RegWeights) -> torch.Tensor:
    """
    Regularization terms for interpretability / identifiability.

    Supported basis keys:
      - symmetric operator: outputs["graph_basis"]            [N,r]
      - directed operator:  outputs["graph_left_basis"]       [N,r], outputs["graph_right_basis"] [N,r]

    Other optional keys:
      - outputs["graph_scales"]   [B,O,r]
      - outputs["fusion_weights"] dict with key "w_graph" (any shape; we take mean(abs(.)))
    """
    # infer dev
    dev = None
    for k, v in outputs.items():
        if torch.is_tensor(v):
            dev = v.device
            break
        if isinstance(v, dict):
            for vv in v.values():
                if torch.is_tensor(vv):
                    dev = vv.device
                    break
    if dev is None:
        dev = torch.device("cpu")

    reg_loss = torch.zeros((), device=dev)

    # ---------- graph basis regularization ----------
    def _basis_regs(B: torch.Tensor) -> torch.Tensor:
        rl = torch.zeros((), device=B.device)
        if reg.reg_graph_orth > 0:
            rdim = B.shape[1]
            BtB = B.T @ B  # [r,r]
            I = torch.eye(rdim, device=B.device, dtype=BtB.dtype)
            rl = rl + float(reg.reg_graph_orth) * torch.mean((BtB - I) ** 2)
        if reg.reg_graph_l1 > 0:
            rl = rl + float(reg.reg_graph_l1) * torch.mean(torch.abs(B))
        return rl

    if "graph_basis" in outputs and torch.is_tensor(outputs["graph_basis"]):
        reg_loss = reg_loss + _basis_regs(outputs["graph_basis"])
    else:
        # directed / two-basis case
        if "graph_left_basis" in outputs and torch.is_tensor(outputs["graph_left_basis"]):
            reg_loss = reg_loss + _basis_regs(outputs["graph_left_basis"])
        if "graph_right_basis" in outputs and torch.is_tensor(outputs["graph_right_basis"]):
            reg_loss = reg_loss + _basis_regs(outputs["graph_right_basis"])

    # ---------- graph scale smoothness across horizons ----------
    if reg.reg_graph_scale_smooth > 0 and "graph_scales" in outputs and torch.is_tensor(outputs["graph_scales"]):
        s = outputs["graph_scales"]  # [B,O,r]
        if s.ndim == 3 and s.shape[1] > 1:
            ds = s[:, 1:, :] - s[:, :-1, :]
            reg_loss = reg_loss + float(reg.reg_graph_scale_smooth) * torch.mean(ds ** 2)

    # ---------- fusion weight sparsity ----------
    # NOTE: L1 on {w_base,w_graph} with w_base+w_graph=1 is constant.
    # Here we interpret reg_fusion_l1 as sparsity penalty on the *graph weight* w_graph only
    # (encourages the model to use spatial correction when it helps generalization).
    if reg.reg_fusion_l1 > 0 and "fusion_weights" in outputs and isinstance(outputs["fusion_weights"], dict):
        fw = outputs["fusion_weights"]
        if "w_graph" in fw and torch.is_tensor(fw["w_graph"]):
            reg_loss = reg_loss + float(reg.reg_fusion_l1) * torch.abs(fw["w_graph"]).float().mean()
        else:
            # fallback: average over all provided tensors
            vals = [v.float().mean() for v in fw.values() if torch.is_tensor(v)]
            if len(vals) > 0:
                reg_loss = reg_loss + float(reg.reg_fusion_l1) * torch.stack(vals).mean()

    return reg_loss
# =========================
# Total loss (for internal forward computation)
# =========================

def compute_total_loss(
    outputs: Dict,
    targets: torch.Tensor,
    targets_mask: Optional[torch.Tensor] = None,
    point_loss: str = "mae",
    huber_delta: float = 1.0,
    lambda_point: float = 1.0,
    lambda_nll: float = 1.0,
    reg_weights: Optional[RegWeights] = None,
    eps: float = 1e-6,
    check_domain: bool = True,
) -> Dict[str, torch.Tensor]:
    if "prediction" not in outputs:
        raise ValueError("outputs must contain 'prediction'.")

    pred = outputs["prediction"]
    lp = pred.new_tensor(0.0)
    if float(lambda_point) > 0:
        lp = compute_point_loss(pred, targets, targets_mask, point_loss=point_loss, huber_delta=huber_delta)

    ln = pred.new_tensor(0.0)
    if float(lambda_nll) > 0 and ("dist_name" in outputs) and ("dist_params" in outputs):
        ln = compute_distribution_loss(
            targets=targets,
            targets_mask=targets_mask,
            dist_name=outputs["dist_name"],
            dist_params=outputs["dist_params"],
            eps=eps,
            check_domain=check_domain,
        )

    reg = compute_regularization(outputs, reg_weights if reg_weights is not None else RegWeights())
    total = float(lambda_point) * lp + float(lambda_nll) * ln + reg

    return {
        "loss": total,
        "loss_point": lp.detach(),
        "loss_nll": ln.detach(),
        "loss_reg": reg.detach(),
    }
