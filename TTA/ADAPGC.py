import time
from copy import deepcopy
import torch
import torch.nn as nn
import torch.jit
from torch.cuda.amp import autocast,GradScaler
import math
import torch.nn.functional as F
from sklearn.decomposition import PCA
from torch_ema import ExponentialMovingAverage
import torch
import torch, gc

def sym_kl_from_logits(z1: torch.Tensor, z2: torch.Tensor, dim: int = -1, eps: float = 1e-12):
    """
    Calculate the symmetric KL divergence (batch form) for two sets of logits.
    """
    if not torch.isfinite(z1).all():
        raise ValueError("z1 contains NaN or Inf")
    if not torch.isfinite(z2).all():
        raise ValueError("z2 contains NaN or Inf")
    p = F.softmax(z1, dim=dim).clamp_min(eps)
    q = F.softmax(z2, dim=dim).clamp_min(eps)
    if not torch.isfinite(p).all():
        raise ValueError("p contains NaN or Inf after softmax")
    if not torch.isfinite(q).all():
        raise ValueError("q contains NaN or Inf after softmax")
    kl_pq = (p * (p.log() - q.log())).sum(dim=dim)
    kl_qp = (q * (q.log() - p.log())).sum(dim=dim)
    return 0.5 * (kl_pq + kl_qp)


class IncrementalGMM:
    """
    Incremental Gaussian Mixture Model with soft-label batch updates and robust Cholesky-based prediction.
    Maintains sufficient statistics (Nk, Sk, Qk) for each component and ensures positive-definite covariances.
    """
    def __init__(self, n_components, n_features, init_mu, init_bias, alpha=0.9, reg_covar=1e-6, dtype=torch.float32, device=None):
        self.K = n_components
        self.d = n_features
        self.alpha = alpha
        self.reg_covar = reg_covar
        self.dtype = dtype
        self.device = device or torch.device('cpu')
        # Initialize parameters
        self.weights_ = torch.ones(self.K, dtype=dtype, device=self.device) / self.K
        
        if init_mu is not None:
            init_mu = init_mu.to(dtype=self.dtype, device=self.device)  # [K, d-1]
            init_bias = init_bias.to(dtype=self.dtype, device=self.device)  # [K]
            assert init_mu.shape == (self.K, self.d), f"init_mu must have shape ({self.K}, {self.d})"
            assert init_bias is not None and init_bias.shape == (self.K,), f"init_bias must have shape ({self.K},)"
            self.means_ = init_mu.to(dtype=self.dtype, device=self.device)
        else:
            assert self.d > 0, "Feature dimension must be positive"
            self.means_ = torch.zeros(self.K, self.d, dtype=dtype, device=self.device)
        eye = torch.eye(self.d, dtype=dtype, device=self.device)
        self.covariances_ = eye.unsqueeze(0).repeat(self.K, 1, 1)
        # Compute robust Cholesky factors
        self._update_cholesky()
        # initialize weights
        self.weights_ = torch.exp(init_bias + 0.5 * self.log_det_cov_ + 0.5 * self.means_.pow(2).sum(dim=1))  # [K]
        
        N_pseudo = 0  # pseudo-samples for each component
        self.Nk = torch.ones(self.K, dtype=dtype, device=self.device) * N_pseudo
        self.Sk = self.Nk.unsqueeze(1) * self.means_
        # self.Qk = torch.einsum('k, kij -> kij', self.Nk, torch.eye(self.d, dtype=dtype, device=self.device).unsqueeze(0).repeat(self.K,1,1))
        # self.Qk += torch.einsum('k, ki, kj -> kij', self.Nk, self.means_, self.means_)
        self.N = self.Nk.sum()
        
    # def _update_cholesky(self):
    #     """
    #     Compute Cholesky decomposition with jitter to ensure positive-definiteness.
    #     """
    #     cov = self.covariances_.clone()
    #     jitter = self.reg_covar
    #     for attempt in range(5):
    #         try:
    #             L = torch.linalg.cholesky(cov)
    #             # success
    #             self.cholesky_L = L
    #             diag = torch.diagonal(L, dim1=-2, dim2=-1)
    #             self.log_det_cov_ = 2.0 * torch.sum(torch.log(diag), dim=-1)
    #             return
    #         except RuntimeError:
    #             cov = cov + jitter * torch.eye(self.d, dtype=self.dtype, device=self.device)
    #             jitter *= 10
    #     raise RuntimeError("Covariance not positive-definite even after adding jitter")
    def _update_cholesky(self):
        if hasattr(self, "cholesky_L"):
            self.cholesky_L = None

        try:
            L = torch.linalg.cholesky(self.covariances_)
        except RuntimeError:
            cov_work = self.covariances_.clone()
            jitter = self.reg_covar
            success = False

            for _ in range(4):
                torch.diagonal(cov_work, dim1=-2, dim2=-1).add_(jitter)

                try:
                    L = torch.linalg.cholesky(cov_work)
                    success = True
                    break
                except RuntimeError:
                    jitter *= 10

            if not success:
                raise RuntimeError("Covariance not positive-definite even after adding jitter")

            del cov_work

        self.cholesky_L = L
        diag = torch.diagonal(self.cholesky_L, dim1=-2, dim2=-1)
        self.log_det_cov_ = 2.0 * torch.sum(torch.log(diag), dim=-1)
    
    @torch.no_grad()
    def update_batch(self, X, gamma):
        """Incremental update with batch X and responsibilities gamma."""
        X = X.to(dtype=self.dtype, device=self.device)
        gamma = gamma.to(dtype=self.dtype, device=self.device)
        B, d = X.shape
        assert d == self.d, f"Feature dimension mismatch: got {d}, expected {self.d}"
        assert gamma.shape == (B, self.K), f"Gamma must have shape (B, {self.K})"

        # Update sufficient statistics
        alpha = self.alpha
        delta_Nk = gamma.sum(dim=0)
        delta_Sk = gamma.t() @ X
        delta_Qk = torch.einsum('bk,bi,bj->kij', gamma, X, X)
        self.Nk += delta_Nk
        self.Sk += delta_Sk
        # self.Qk += delta_Qk
        self.N += delta_Nk.sum()
        # Recompute parameters
        means = self.means_
        self.weights_ = alpha * self.weights_ + (1 - alpha) * self.Nk / self.N
        self.means_ = alpha * self.means_ + (1 - alpha) * self.Sk / self.Nk.unsqueeze(1)
        
        means = self.means_
        for k in range(self.K):
            self.covariances_[k] = alpha * self.covariances_[k] + (1 - alpha) * (
                ((self.covariances_[k] + torch.outer(means[k], means[k])) * (self.Nk[k]) + delta_Qk[k])/ self.Nk[k] - torch.outer(self.means_[k], self.means_[k])  # [d, d]
            )
        self._update_cholesky()
        
    def predict_batch(self, X, device=None):
        X = X.to(dtype=self.dtype, device=self.device)
        B, d = X.shape

        const = -0.5 * d * torch.log(torch.tensor(2 * torch.pi, dtype=self.dtype, device=self.device))

        # diff: [K, B, d]
        diff = X.unsqueeze(0) - self.means_.unsqueeze(1)

        # rhs: [K, d, B]
        # For each class k, solve:
        #   Sigma_k^{-1} (X - mu_k)^T
        rhs = diff.transpose(1, 2)

        # self.cholesky_L: [K, d, d]
        # sol: [K, d, B]
        sol = torch.cholesky_solve(rhs, self.cholesky_L, )

        # Mahalanobis distance:
        #   (x - mu)^T Sigma^{-1} (x - mu)
        # rhs and sol are both [K, d, B]
        # maha: [K, B] -> [B, K]
        maha = (rhs * sol).sum(dim=1).transpose(0, 1)

        logits = (
            torch.log(self.weights_.clamp_min(1e-12)).unsqueeze(0)
            - 0.5 * self.log_det_cov_.unsqueeze(0)
            - 0.5 * maha
            + const
        )

        if device is not None:
            logits = logits.to(device)

        return logits
        # logits_list = []

        # for k in range(self.K):
        #     L_k = self.cholesky_L[k]              # [d, d]
        #     mu_k = self.means_[k]                 # [d]
        #     diff = X - mu_k.unsqueeze(0)          # [B, d]

        #     diff_vec = diff.unsqueeze(-1)         # [B, d, 1]

        #     # solve Sigma_k^{-1} diff
        #     sol = torch.cholesky_solve(
        #         diff_vec,
        #         L_k.unsqueeze(0).expand(B, -1, -1)
        #     )                                    # [B, d, 1]

        #     maha = torch.matmul(
        #         diff_vec.transpose(-2, -1),
        #         sol
        #     ).squeeze(-1).squeeze(-1)             # [B]

        #     logit_k = (
        #         torch.log(self.weights_[k].clamp_min(1e-12))
        #         - 0.5 * self.log_det_cov_[k]
        #         - 0.5 * maha
        #         + const
        #     )                                    # [B]

        #     logits_list.append(logit_k)

        # logits = torch.stack(logits_list, dim=1)  # [B, K]

        # if device is not None:
        #     logits = logits.to(device)

        # return logits
    
    def get_params(self):
        """Return weights, means, covariances."""
        return self.weights_, self.means_, self.covariances_
    
    def dispose(self):
        """Release GPU/CPU memory occupied by this GMM (safe to call multiple times)."""
        # 1) If on GPU, synchronize this device first
        try:
            if isinstance(self.device, torch.device) and self.device.type == 'cuda':
                with torch.cuda.device(self.device):
                    torch.cuda.synchronize()
        except Exception:
            pass

        # 2) Set attributes to None (break Python reference chain so tensors can be freed)
        for name in [
            'Nk','Sk','Qk','weights_','means_','covariances_',
            'cholesky_L','log_det_cov_'
        ]:
            if hasattr(self, name):
                try:
                    setattr(self, name, None)
                except Exception:
                    pass

        # 3) Force garbage collection
        gc.collect()

        # 4) Clear CUDA caches on this device
        try:
            if isinstance(self.device, torch.device) and self.device.type == 'cuda':
                with torch.cuda.device(self.device):
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
                    torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass




def init_cross_stats(gmm_a, gmm_v, gmm_f):
# def init_cross_stats(K, dim, device, dtype):
    
    device = gmm_f.device
    dtype = gmm_f.dtype

    K = gmm_f.K
    dim = gmm_a.d

    eye = torch.eye(dim, dtype=dtype, device=device).unsqueeze(0).repeat(K, 1, 1)

    cross_stats = {
        "Nk": torch.zeros(K, dtype=dtype, device=device),
        # "Q_AF": torch.zeros(K, dim_a, dim_f, dtype=dtype, device=device),
        # "Q_VF": torch.zeros(K, dim_v, dim_f, dtype=dtype, device=device),
        "mean_A": gmm_a.means_.detach().clone(),
        "mean_V": gmm_v.means_.detach().clone(),
        "mean_F": gmm_f.means_.detach().clone(),
        # eye = torch.eye(self.d, dtype=dtype, device=self.device)
        # self.covariances_ = torch.eye(self.d, dtype=dtype, device=self.device).unsqueeze(0).repeat(self.K, 1, 1)
        "Sigma_AF": eye.clone(),
        "Sigma_VF":  eye.clone(),
    }

    return cross_stats


@torch.no_grad()
def update_cross_stats_from_full(cross_stats, gmm_a, gmm_v, gmm_f, ca, cv, feat_full, gamma_full, alpha):
    device = gmm_f.device
    dtype = gmm_f.dtype
    K = gmm_f.K

    ca = ca.detach().to(dtype=dtype, device=device)
    cv = cv.detach().to(dtype=dtype, device=device)
    feat_full = feat_full.detach().to(dtype=dtype, device=device)
    gamma_full = gamma_full.detach().to(dtype=dtype, device=device)

    delta_Q_AF = torch.einsum("bk,bi,bj->kij", gamma_full, ca, feat_full, )  # [K, dim_a, dim_f]
    delta_Q_VF = torch.einsum("bk,bi,bj->kij", gamma_full, cv, feat_full, )  # [K, dim_v, dim_f]

    old_Nk = cross_stats["Nk"]                  # [K]
    delta_Nk = gamma_full.sum(dim=0)            # [K]
    new_Nk = old_Nk + delta_Nk                  # [K]
        
    old_mean_A = cross_stats["mean_A"]
    old_mean_V = cross_stats["mean_V"]
    old_mean_F = cross_stats["mean_F"]

    for k in range(K):
        cross_stats["Sigma_AF"][k] = alpha * cross_stats["Sigma_AF"][k] + (1 - alpha) *(
            ((cross_stats["Sigma_AF"][k] + torch.outer(old_mean_A[k], old_mean_F[k])) * new_Nk[k] + delta_Q_AF[k]) 
            / new_Nk[k] 
            - torch.outer(gmm_a.means_[k], gmm_f.means_[k])
        )
        cross_stats["Sigma_VF"][k] = alpha * cross_stats["Sigma_VF"][k] + (1 - alpha) *(
            ((cross_stats["Sigma_VF"][k] + torch.outer(old_mean_V[k], old_mean_F[k])) * new_Nk[k] + delta_Q_VF[k]) 
            / new_Nk[k] 
            - torch.outer(gmm_v.means_[k], gmm_f.means_[k])
        )

    cross_stats["Nk"] = new_Nk
    cross_stats["mean_A"] = gmm_a.means_.detach().clone()
    cross_stats["mean_V"] = gmm_v.means_.detach().clone()
    cross_stats["mean_F"] = gmm_f.means_.detach().clone()
        

def conditional_mean_f_given_x(cross_stats, gmm_x, gmm_f, f_x, source):
    """
    Compute class-wise conditional full-view means:
        mu_k^{F|X}(f_X)
        =
        mu_k^F + Sigma_k^{FX} (Sigma_k^X)^{-1} (f_X - mu_k^X)
    where X can be:
        source="a": X = A
        source="v": X = V
    """
    device = gmm_f.device
    dtype = gmm_f.dtype

    f_x = f_x.to(dtype=dtype, device=device)

    if source == "a":
        sigma_xf_key = "Sigma_AF"
    elif source == "v":
        sigma_xf_key = "Sigma_VF"
    else:
        raise ValueError(f"Unknown source: {source}")
    
    if f_x.dim() == 1:
        f_x = f_x.unsqueeze(0)

    expected_dim = gmm_x.d

    B, dim_x = f_x.shape
    K = gmm_f.K
    dim_f = gmm_f.d

    cond_means = torch.zeros(B, K, dim_f, dtype=dtype, device=device, )

    eye_x = torch.eye(dim_x, dtype=dtype, device=device, )

    eps = 1e-12

    for k in range(K):
        # If no paired full samples have been observed for class k,
        # fall back to the full-view class mean.
        if cross_stats["Nk"][k] <= eps:
            cond_means[:, k, :] = gmm_f.means_[k].unsqueeze(0)
            continue

        mu_x_k = gmm_x.means_[k]  # [dim_f]
        mu_f_k = gmm_f.means_[k]  # [dim_f]
        # same = torch.allclose(mu_x_k, mu_f_k, rtol=1e-5, atol=1e-8)
        # print(same)
        # Sigma_XF[k] = Cov(f_X, f_F | k), shape [dim_x, dim_f]
        Sigma_XF_k = cross_stats[sigma_xf_key][k]
        # Sigma_FX[k] = Cov(f_F, f_X | k), shape [dim_f, dim_x]
        Sigma_FX_k = Sigma_XF_k.transpose(0, 1)
        # print(is_identity_matrix(Sigma_XF_k))
        # print(is_identity_matrix(Sigma_FX_k))

        cov_x_k = gmm_x.covariances_[k] + gmm_x.reg_covar * eye_x

        L_x = torch.linalg.cholesky(cov_x_k)

        diff_x = f_x - mu_x_k.unsqueeze(0)  # [B, dim_x]

        sol_x = torch.cholesky_solve(diff_x.transpose(0, 1), L_x, )  # [dim_x, B]

        delta_f = Sigma_FX_k @ sol_x  # [dim_f, B]
        
        cond_means[:, k, :] = (mu_f_k.unsqueeze(0) + delta_f.transpose(0, 1))  # [B, K, dim_f]

    return cond_means


# @torch.no_grad()
# def predict_x2f(cross_stats, gmm_x, gmm_f, f_x, source, temp=20, device=None):
#     """
#     Plug-in prediction for missing-modality samples:
#         p_{X->F}(c | f_X)
#         approx sum_k p_X(k | f_X) p_F(c | mu_k^{F|X}(f_X))
#     where X can be:
#         source="a": audio-only feature f_A, used for A -> F
#         source="v": video-only feature f_V, used for V -> F
#     """
#     device_out = device
#     device = gmm_f.device
#     dtype = gmm_f.dtype

#     f_x = f_x.to(dtype=dtype, device=device)

#     # If no paired full samples have been observed,
#     # fall back to direct current-view GMM prediction.
#     if cross_stats["Nk"].sum() <= 0:
#         logits_x = gmm_x.predict_batch(f_x, device=device)
#         # log_probs = torch.log_softmax(logits_x, dim=-1)

#         if device_out is not None:
#             # log_probs = log_probs.to(device_out)
#             logits_x = logits_x.to(device_out)

#         # return log_probs
#         return logits_x

#     logits_x = gmm_x.predict_batch(f_x, device=device)
#     alpha_x = torch.softmax(logits_x / temp, dim=-1)  # [B, K]

#     cond_means = conditional_mean_f_given_x(cross_stats=cross_stats, gmm_x=gmm_x, gmm_f=gmm_f, f_x=f_x, source=source, )  # [B, K, dim_f]

#     B, K, dim_f = cond_means.shape

#     # probs = torch.zeros(B, K, dtype=dtype, device=device, )
#     cond_flat = cond_means.reshape(B * K, dim_f)
#     logits_f_all = gmm_f.predict_batch(cond_flat, device=device)
#     logits_f_all = logits_f_all.view(B, K, K)

#     probs_f_all = torch.softmax(logits_f_all, dim=-1)
#     # probs = (alpha_x.unsqueeze(-1) * probs_f_all).sum(dim=1)
#     probs = (alpha_x.unsqueeze(-1) * logits_f_all).sum(dim=1)

#     if device_out is not None:
#         probs = probs.to(device_out)
        
#     return probs


# @torch.no_grad()
# def recovery_uncertainty_penalty(cross_stats, gmm_x, gmm_f, source, score_block=8):
#     """
#     Exact low-memory computation of

#         penalty[c, k]
#         = 0.5 * Tr(
#             Sigma_{F,k}^{-1} Sigma_{F|X,c}
#           )

#     where

#         Sigma_{F|X,c}
#         = Sigma_{F,c}
#           - Sigma_{FX,c}
#             Sigma_{X,c}^{-1}
#             Sigma_{XF,c}.

#     Args:
#         cross_stats:
#             Online cross-view statistics.

#         gmm_x:
#             GMM of the observed modality X.

#         gmm_f:
#             GMM of the fused view F.

#         source:
#             "a" for A -> F,
#             "v" for V -> F.

#         score_block:
#             Number of scoring classes k processed at once.
#             Only affects memory/speed, NOT the result.

#     Returns:
#         penalty: [K, K]

#             penalty[c, k] corresponds to

#             0.5 * Tr(
#                 Sigma_{F,k}^{-1}
#                 Sigma_{F|X,c}
#             )
#     """

#     device = gmm_f.device
#     dtype = gmm_f.dtype

#     if source == "a":
#         cross_xf = cross_stats["Sigma_AF"]
#     elif source == "v":
#         cross_xf = cross_stats["Sigma_VF"]
#     else:
#         raise ValueError(f"Unknown source: {source}")

#     cross_xf = cross_xf.to(device=device, dtype=dtype)
#     K = gmm_f.covariances_.shape[0]

#     # Only K x K scalars are kept permanently.
#     penalty = torch.empty(K, K, dtype=dtype, device=device)

#     # ---------------------------------------------------------
#     # c: class used for conditional recovery
#     # ---------------------------------------------------------
#     for c in range(K):

#         # Sigma_X,c^{-1} Sigma_XF,c
#         #
#         # cross_xf[c] = Sigma_XF,c
#         solved = torch.cholesky_solve(cross_xf[c], gmm_x.cholesky_L[c])

#         # Sigma_F|X,c
#         # =
#         # Sigma_F,c
#         # - Sigma_FX,c Sigma_X,c^{-1} Sigma_XF,c
#         cond_cov = gmm_f.covariances_[c] - cross_xf[c].transpose(-1, -2) @ solved

#         # Remove tiny numerical asymmetry.
#         cond_cov = 0.5 * (cond_cov + cond_cov.transpose(-1, -2))

#         # -----------------------------------------------------
#         # k: class used by fused-view GDA for scoring
#         #
#         # Process k in blocks to limit GPU memory.
#         # -----------------------------------------------------
#         for start in range(0, K, score_block):
#             end = min(start + score_block, K)

#             # expand() creates a view rather than physically
#             # copying cond_cov score_block times.
#             rhs = cond_cov.unsqueeze(0).expand(end - start, -1, -1)

#             # For each k in this block:
#             #
#             # tmp[k]
#             # = Sigma_F,k^{-1} Sigma_F|X,c
#             tmp = torch.cholesky_solve(rhs, gmm_f.cholesky_L[start:end])

#             # Tr(Sigma_F,k^{-1} Sigma_F|X,c)
#             penalty[c, start:end] = (
#                 0.5 * torch.diagonal(tmp, dim1=-2, dim2=-1).sum(dim=-1)
#             )

#     return penalty



@torch.no_grad()
def recovery_uncertainty_penalty(cross_stats, gmm_x, gmm_f, source):
    device = gmm_f.device
    dtype = gmm_f.dtype

    if source == "a":
        cross_xf = cross_stats["Sigma_AF"]
    elif source == "v":
        cross_xf = cross_stats["Sigma_VF"]
    else:
        raise ValueError(f"Unknown source: {source}")

    cross_xf = cross_xf.to(device=device, dtype=dtype,)

    cross_diag = torch.diagonal(cross_xf, dim1=-2, dim2=-1,)  # [K,D]

    var_x_effective = torch.einsum("kij,kij->ki", gmm_x.cholesky_L, gmm_x.cholesky_L)  # [K,D]

    var_f_score_effective = torch.einsum("kij,kij->ki", gmm_f.cholesky_L, gmm_f.cholesky_L)  # [K,D]

    var_f_diag = torch.diagonal(gmm_f.covariances_, dim1=-2, dim2=-1)  # [K,D]

    cond_diag = (var_f_diag - cross_diag.square() / var_x_effective)  # [K,D]

    inverse_score_var = (var_f_score_effective.reciprocal())  # [K,D]

    penalty = 0.5 * (cond_diag @ inverse_score_var.transpose(0, 1))  # [K,K]
    
    # raw_cond_diag = (
    #     var_f_diag
    #     - cross_diag.square()
    #     / var_x_effective.clamp_min(1e-8)
    # )

    # print(
    #     source,
    #     "negative ratio:", (raw_cond_diag < 0).float().mean().item(),
    #     "min:", raw_cond_diag.min().item(),
    #     "max:", raw_cond_diag.max().item(),
    # )

    return penalty


@torch.no_grad()
def predict_x2f(cross_stats, gmm_x, gmm_f, f_x, source, temp=20, warmup=50, device=None):
    """
    Uncertainty-aware prediction for missing-modality samples.

    For each recovery class c and scoring class k:

        E[g_{F,k}(z_F) | z_X, y=c]
        =
        g_{F,k}(hat{z}_{F|X,c})
        - 0.5 Tr(
            Sigma_{F,k}^{-1}
            Sigma_{F|X,c}
          )
    """

    device_out = device
    device = gmm_f.device
    dtype = gmm_f.dtype

    f_x = f_x.to(dtype=dtype, device=device)

    # No paired full samples have been observed.
    if cross_stats["Nk"].sum() <= warmup:
        logits_x = gmm_x.predict_batch(f_x, device=device)
        if device_out is not None:
            logits_x = logits_x.to(device_out)
        return logits_x

    # ---------------------------------------------------------
    # 1. Prediction from the observed modality
    # ---------------------------------------------------------
    logits_x = gmm_x.predict_batch(f_x, device=device)
    alpha_x = torch.softmax(logits_x / temp, dim=-1)  # [B, K]

    # ---------------------------------------------------------
    # 2. Conditional means
    #
    #    hat{z}_{F|X,c}
    # ---------------------------------------------------------
    cond_means = conditional_mean_f_given_x(
        cross_stats=cross_stats, gmm_x=gmm_x, gmm_f=gmm_f,
        f_x=f_x, source=source,
    )  # [B, K, dim_f]

    B, K, dim_f = cond_means.shape

    # ---------------------------------------------------------
    # 3. GDA scores at the recovered means
    #
    # logits_f_all[b, c, k]
    # =
    # g_{F,k}(hat{z}_{F|X,c})
    # ---------------------------------------------------------
    cond_flat = cond_means.reshape(B * K, dim_f)
    logits_f_all = gmm_f.predict_batch(cond_flat, device=device).view(B, K, K)

    # ---------------------------------------------------------
    # 4. Exact uncertainty penalty
    #
    # penalty[c, k]
    # =
    # 0.5 Tr(
    #     Sigma_F,k^{-1}
    #     Sigma_F|X,c
    # )
    # ---------------------------------------------------------
    # penalty = recovery_uncertainty_penalty(
    #     cross_stats, gmm_x, gmm_f, source
    # )  # [K, K]

    # # Expected GDA score
    # logits_f_all = logits_f_all - penalty.unsqueeze(0)

    # ---------------------------------------------------------
    # 5. Marginalize over the unknown recovery class c
    # ---------------------------------------------------------
    logits = (alpha_x.unsqueeze(-1) * logits_f_all).sum(dim=1)  # [B, K]

    if device_out is not None:
        logits = logits.to(device_out)

    return logits




def entropy_weighted_logits(logits1, logits2, tau=10):
    # probs: [bs, K]
    p1 = F.softmax(logits1, dim=-1)
    p2 = F.softmax(logits2, dim=-1)

    # entropy: [bs]
    eps = 1e-8
    H1 = -(p1 * torch.log(p1 + eps)).sum(dim=-1)
    H2 = -(p2 * torch.log(p2 + eps)).sum(dim=-1)

    # lower entropy -> larger weight
    scores = torch.stack([-H1 / tau, -H2 / tau], dim=-1)  # [bs, 2]
    weights = F.softmax(scores, dim=-1)                   # [bs, 2]

    w1 = weights[:, 0:1]  # [bs, 1]
    w2 = weights[:, 1:2]  # [bs, 1]

    fused_logits = w1 * logits1 + w2 * logits2
    return fused_logits, weights

def sample_count_weighted_logits(logits1, logits2, learned_num, N0=5000):
    """
    Args:
        logits1: Tensor, shape [bs, K]
        logits2: Tensor, shape [bs, K]
        learned_num: 当前已经学到的样本数，可以是 int / float / scalar tensor
        N0: 唯一超参数。含义：学到 N0 个样本后，完全信任 logits2。

    Returns:
        fused_logits: Tensor, shape [bs, K]
        weight2: Tensor scalar, logits2 的权重
    """

    assert logits1.shape == logits2.shape, "logits1 and logits2 must have the same shape"

    device = logits1.device
    dtype = logits1.dtype

    learned_num = torch.as_tensor(learned_num, device=device, dtype=dtype)
    N0 = torch.as_tensor(N0, device=device, dtype=dtype)

    weight2 = torch.clamp(learned_num / N0, min=0.0, max=1.0)
    weight1 = 1.0 - weight2

    fused_logits = weight1 * logits1 + weight2 * logits2

    return fused_logits, weight2

@torch.no_grad()
def recovery_reliability(cholesky_m, cov_f, cross_mf, device_out=None, eps: float = 1e-8,):
    """
    Low-memory implementation.

    cholesky_m: [C, d, d]
    cov_f:      [C, d, d]
    cross_mf:   [C, d, d]
    """

    C = cholesky_m.shape[0]
    rho = torch.empty(C, dtype=cholesky_m.dtype, device=cholesky_m.device,)

    for c in range(C):

        # Only creates one d x d temporary tensor
        Y = torch.linalg.solve_triangular(cholesky_m[c],
            cross_mf[c],
            upper=False,
        )

        explained_var = Y.square().sum()

        total_var = torch.diagonal(cov_f[c]).sum()

        rho[c] = explained_var / total_var.clamp_min(eps)
        # print(rho[c])
    if device_out is not None:
        rho = rho.to(device_out)

    return rho.clamp_(0.0, 1.0)


@torch.no_grad()
def recovery_uncertainty_weight( cholesky_m, cov_f, cross_mf, device_out=None,):
    """
    Compute class-wise recovery weights based on conditional uncertainty.

    Args:
        cholesky_m: [C, d, d]
            Cholesky factors of Sigma_m:
            Sigma_m = L_m @ L_m.T

        cov_f: [C, d, d]
            Covariance matrices of the fused view.

        cross_mf: [C, d, d]
            Cross-covariance Sigma_mF.

        device_out:
            Device of returned weights. If None, keep the original device.

    Returns:
        weight: [C]
            Class-wise dynamic recovery weights.
    """

    C, d, _ = cholesky_m.shape

    # Store only C scalars.
    uncertainty = torch.empty(C, dtype=torch.float32, device=cholesky_m.device,)

    # Fixed internally to keep temporary memory very small.
    chunk_size = 32
    eps = 1e-8

    for c in range(C):
        explained_var = torch.zeros(
            (),
            dtype=torch.float32,
            device=cholesky_m.device,
        )

        # Compute ||L_m^{-1} Sigma_mF||_F^2 by column chunks,
        # avoiding a full d x d temporary result.
        for start in range(0, d, chunk_size):
            end = min(start + chunk_size, d)

            rhs = cross_mf[c, :, start:end]

            Y = torch.linalg.solve_triangular(
                cholesky_m[c],
                rhs,
                upper=False,
            )

            explained_var += Y.float().square().sum()

        # Tr(Sigma_F)
        total_var = torch.diagonal(
            cov_f[c],
            dim1=-2,
            dim2=-1,
        ).float().sum().clamp_min(eps)

        # Explained-variance ratio
        rho = (explained_var / total_var).clamp(0.0, 1.0)

        # Normalized conditional uncertainty
        uncertainty[c] = 1.0 - rho

    # Typical conditional uncertainty of this observed modality.
    mean_uncertainty = uncertainty.mean()

    # When u_c == mean(u), weight = 0.5.
    weight = mean_uncertainty / (
        mean_uncertainty + uncertainty + eps
    )

    weight = weight.clamp_(0.0, 1.0)

    if device_out is not None:
        weight = weight.to(device_out)

    return weight

class ADAPGC(nn.Module):
    """Tent adapts a model by entropy minimization during testing.
    Once tented, a model adapts itself by updating on every forward.
    """
    def __init__(self, model, optimizer, device, args, steps=1, episodic=False):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.steps = steps
        assert steps > 0, "tent requires >= 1 step(s) to forward and update"
        self.episodic = episodic
        self.args = args
        self.scaler = GradScaler()
        self.device = device
        init_mu = model.module.mlp_head[-1].weight.detach().clone()  # [K, D]
        init_bias = model.module.mlp_head[-1].bias.detach().clone()  # [K]
        
        gmm_device = torch.device("cuda:0" if torch.cuda.device_count() <= 1 else "cuda:1")
        device = gmm_device
        self.gmm_f = IncrementalGMM(n_components=init_mu.size(0), n_features=init_mu.size(1), init_mu=init_mu, init_bias=init_bias, reg_covar=1e-6, device=device, dtype=torch.float32)
        self.gmm_a = IncrementalGMM(n_components=init_mu.size(0), n_features=init_mu.size(1), init_mu=init_mu, init_bias=init_bias, reg_covar=1e-6, device=device, dtype=torch.float32)
        self.gmm_v = IncrementalGMM(n_components=init_mu.size(0), n_features=init_mu.size(1), init_mu=init_mu, init_bias=init_bias, reg_covar=1e-6, device=device, dtype=torch.float32)
        self.cross_stats = init_cross_stats(self.gmm_a, self.gmm_v, self.gmm_f)
        self.ema = ExponentialMovingAverage(self.model.parameters(), decay=0.999)
        ln_params = collect_ln_params(self.model)
        print('requires_grad:', sum(p.requires_grad for p in ln_params))
        print('len:', len(ln_params))
        self.ln_optimizer = torch.optim.Adam([{'params': ln_params, 'lr': 1e-4}], weight_decay=0., betas=(0.9, 0.999))

    def forward(self, x, adapt_flag):
        for _ in range(self.steps):
            if adapt_flag:
                outputs, loss = forward_and_adapt(
                    x, self.model, self.optimizer, self.args, self.scaler, self.ema, 
                    self.gmm_f, self.gmm_a, self.gmm_v, self.cross_stats, self.ln_optimizer
                )
                # outputs, loss = forward_and_adapt(x, self.model, self.optimizer, self.args, self.scaler, self.ema, self.gmm_f, self.gmm_a, self.gmm_v, self.ln_optimizer)
            else:
                outputs, _ = self.model.module.forward_eval(a=x[0], v=x[1], mode=self.args.testmode)
                loss = (0, 0)
                outputs = (outputs, outputs)

        return outputs, loss
    
    def delete_gmm(self):
        """Release GPU/CPU memory occupied by all GMM instances."""
        self.gmm_f.dispose()
        self.gmm_a.dispose()
        self.gmm_v.dispose()


@torch.jit.script
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)


@torch.enable_grad()  # ensure grads in possible no grad context for testing
def forward_and_adapt(
    x, model, optimizer, args, scaler, ema, 
    gmm_f, gmm_a, gmm_v, cross_stats, ln_optimizer
):
    """Forward and adapt model on batch of data.
    Compute loss function (Eq. 7) based on the model prediction, take gradients, and update params.
    """
    with autocast():
        # forward
        results = model.module.forward_eval_with_features(a=x[0], v=x[1])
        outputs = results['logits']
        ca = results['ca']
        cv = results['cv']
        feat = results['feat']
        mask = results['mask']
        mask_full = mask['full']
        mask_audio = mask['audio_only']
        mask_video = mask['video_only']

    p_sum = outputs.softmax(dim=-1).sum(dim=0)
    loss_bal = - (p_sum.softmax(dim=0) * p_sum.log_softmax(dim=0)).sum()    

    pred = outputs.softmax(dim=-1)
    pred_max = pred.max(dim=-1)[0]
    gamma = math.exp(-1)
    t = torch.ones(outputs.shape[0], device=outputs.device) * gamma
    loss_ra = (pred_max * (1 - pred_max.log() + t.log())).mean()

    # Update Distribution Params
    gmm_f.update_batch(feat, pred)

    if mask_full.sum() > 0:
        gmm_a.update_batch(ca, pred[mask_full])
        gmm_v.update_batch(cv, pred[mask_full])
        update_cross_stats_from_full(
            cross_stats=cross_stats, gmm_a=gmm_a, gmm_v=gmm_v, gmm_f=gmm_f,
            ca=ca, cv=cv, feat_full=feat[mask_full], gamma_full=pred[mask_full], alpha=args.alpha
        )

    if mask_audio.sum() > 0:
        gmm_a.update_batch(feat[mask_audio], pred[mask_audio])

    if mask_video.sum() > 0:
        gmm_v.update_batch(feat[mask_video], pred[mask_video])

    # Prediction from distribution
    logits_gmm = torch.zeros_like(outputs, dtype=torch.float32)

    if mask_full.sum() > 0:
        logits_f = gmm_f.predict_batch(feat[mask_full], device=pred.device)
        logits_a = gmm_a.predict_batch(ca, device=pred.device)
        logits_v = gmm_v.predict_batch(cv, device=pred.device)

        logits_gmm[mask_full] = logits_f

    if mask_audio.sum() > 0:
        # weight =  recovery_uncertainty_weight(gmm_a.cholesky_L, gmm_f.covariances_, cross_stats["Sigma_AF"], device_out=pred.device)
        # logits_gmm[mask_audio] = weight * predict_x2f(cross_stats=cross_stats, gmm_x=gmm_a, gmm_f=gmm_f, f_x=feat[mask_audio],
        #             source="a", temp=args.temp, device=pred.device,).float() + (1 - weight) * gmm_a.predict_batch(feat[mask_audio], device=pred.device)
        logits_recovery = predict_x2f(cross_stats=cross_stats, gmm_x=gmm_a, gmm_f=gmm_f, f_x=feat[mask_audio],
                    source="a", temp=args.temp, warmup=args.warmup_a, device=pred.device,).float()
        logits_gda = gmm_a.predict_batch(feat[mask_audio], device=pred.device)
        logits_gmm[mask_audio] = (1 - args.beta) * logits_recovery + args.beta * logits_gda
        # logits_gmm[mask_audio] = (1 - args.beta) * predict_x2f(cross_stats=cross_stats, gmm_x=gmm_a, gmm_f=gmm_f, f_x=feat[mask_audio],
        #             source="a", temp=args.temp, warmup=args.warmup_a, device=pred.device,).float() + args.beta * gmm_a.predict_batch(feat[mask_audio], device=pred.device)
        
    if mask_video.sum() > 0:
        # weight =  recovery_uncertainty_weight(gmm_v.cholesky_L, gmm_f.covariances_, cross_stats["Sigma_VF"], pred.device)
        # logits_gmm[mask_video] = weight * predict_x2f(cross_stats=cross_stats, gmm_x=gmm_v, gmm_f=gmm_f, f_x=feat[mask_video],
        #             source="v", temp=args.temp, device=pred.device,).float() + (1 - weight) * gmm_v.predict_batch(feat[mask_video], device=pred.device)
        logits_recovery = predict_x2f(cross_stats=cross_stats, gmm_x=gmm_v, gmm_f=gmm_f, f_x=feat[mask_video],
                    source="v", temp=args.temp, warmup=args.warmup_v, device=pred.device,).float() 
        logits_gda = gmm_v.predict_batch(feat[mask_video], device=pred.device)
        logits_gmm[mask_video] = (1 - args.beta) * logits_recovery + args.beta * logits_gda
        # logits_gmm[mask_video] = (1 - args.beta) * predict_x2f(cross_stats=cross_stats, gmm_x=gmm_v, gmm_f=gmm_f, f_x=feat[mask_video],
        #     source="v", temp=args.temp, warmup=args.warmup_v, device=pred.device,).float() + args.beta * gmm_v.predict_batch(feat[mask_video], device=pred.device)

    # Contrastive Loss
    loss_c = None
    loss_gmm = None
    if mask_full.sum() > 0:
        D_a = sym_kl_from_logits(logits_f, logits_a)
        D_v = sym_kl_from_logits(logits_f, logits_v)

        mask_v2a = (D_a < D_v)
        mask_a2v = (D_v <= D_a)

        tau = 0.05
        a_f = F.normalize(ca, p=2, dim=-1)
        v_f = F.normalize(cv, p=2, dim=-1)

        v_f_t = v_f.detach()
        a_f_t = a_f.detach()

        idx_v2a = mask_v2a.nonzero(as_tuple=False).squeeze(-1)
        idx_a2v = mask_a2v.nonzero(as_tuple=False).squeeze(-1)

        loss_parts = []
        num_terms = 0

        if idx_v2a.numel() > 0:
            logits_va = v_f.index_select(0, idx_v2a) @ a_f_t.t() / tau
            labels_va = idx_v2a
            loss_va = F.cross_entropy(logits_va, labels_va, reduction='sum')
            loss_parts.append(loss_va)
            num_terms += idx_v2a.numel()

        if idx_a2v.numel() > 0:
            logits_av = a_f.index_select(0, idx_a2v) @ v_f_t.t() / tau
            labels_av = idx_a2v
            loss_av = F.cross_entropy(logits_av, labels_av, reduction='sum')
            loss_parts.append(loss_av)
            num_terms += idx_a2v.numel()

        if num_terms > 0:
            loss_c = torch.stack(loss_parts).sum() / num_terms


    p = logits_gmm.softmax(dim=1)
    log_q = torch.log_softmax(outputs, dim=1)
    loss_gmm = -(p * log_q).sum(dim=1).mean()

    loss = args.w_read * (loss_ra  - loss_bal )
    
    if loss_gmm is not None:
        loss += args.w_g * loss_gmm
    
    optimizer.zero_grad(set_to_none=True)
    scaler.scale(loss).backward(retain_graph=True)
    scaler.step(optimizer)

    # Update ln_optimizer using only loss_c
    
    ln_optimizer.zero_grad(set_to_none=True)
    if loss_c is not None:
        loss_c = args.w_c * loss_c
        scaler.scale(loss_c).backward()
        scaler.step(ln_optimizer)

    scaler.update()

    ema.update(model.parameters())
    
    ema.store()
    ema.copy_to(model.parameters())

    with torch.no_grad():
        with autocast():
            results = model.module.forward_eval_with_features(a=x[0], v=x[1])
            outputs2 = results['logits']
            # outputs2, _ = model.module.forward_eval(a=x[0], v=x[1], mode=args.testmode)
            
            outputs2 += args.gamma * logits_gmm
    ema.restore()

    return (outputs, outputs2), (loss_ra.item(), loss_bal.item())


def collect_params(model):
    """
    Walk the model's modules and collect qkv parameters of the fusion attn module.
    Return the parameters and their names.
    Note: other choices of parameterization are possible!
    """
    params_fusion_qkv = []
    names_fusion_qkv = []

    for nm, m in model.named_modules():
        if nm == 'module.blocks_u.0.attn.q' or nm == 'module.blocks_u.0.attn.k' or nm == 'module.blocks_u.0.attn.v':
            for np, p in m.named_parameters():
                if np in ['weight', 'bias']:
                    params_fusion_qkv.append(p)
                    names_fusion_qkv.append(f"{nm}.{np}")

    return params_fusion_qkv, names_fusion_qkv


def collect_ln_params(model):
    """Collect the affine scale + shift parameters from batch norms.

    Walk the model's modules and collect all batch normalization parameters.
    Return the parameters and their names.

    Note: other choices of parameterization are possible!
    """
    params = []
    names = []
    for nm, m in model.named_modules():
        if isinstance(m, nn.LayerNorm):
            for np, p in m.named_parameters():
                if np in ['weight', 'bias']:  # weight is scale, bias is shift
                    params.append(p)
                    names.append(f"{nm}.{np}")
    return params


def copy_model_and_optimizer(model, optimizer):
    """Copy the model and optimizer states for resetting after adaptation."""
    model_state = deepcopy(model.state_dict())
    optimizer_state = deepcopy(optimizer.state_dict())
    return model_state, optimizer_state


def load_model_and_optimizer(model, optimizer, model_state, optimizer_state):
    """Restore the model and optimizer states from copies."""
    model.load_state_dict(model_state, strict=True)
    optimizer.load_state_dict(optimizer_state)


def configure_model(model):
    """Configure model for use with Renata."""
    # train mode, but no grad
    model.train()
    model.requires_grad_(False)

    for nm, m in model.named_modules():
        if nm == 'module.blocks_u.0.attn.q' or nm == 'module.blocks_u.0.attn.k' or nm == 'module.blocks_u.0.attn.v':
            m.requires_grad_(True)
    for m in model.modules():
        if isinstance(m, nn.LayerNorm):
            m.requires_grad_(True)

    return model
