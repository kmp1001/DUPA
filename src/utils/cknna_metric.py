# CKNNA 与特征预处理，与 VFM_VAE/tools/evaluate_alignment/compute.py 保持一致
import torch
import torch.nn.functional as F


def hsic_unbiased(K, L):
    m = K.shape[0]
    K_tilde = K.clone().fill_diagonal_(0)
    L_tilde = L.clone().fill_diagonal_(0)

    HSIC_value = (
        (torch.sum(K_tilde * L_tilde.T))
        + (torch.sum(K_tilde) * torch.sum(L_tilde) / ((m - 1) * (m - 2)))
        - (2 * torch.sum(torch.mm(K_tilde, L_tilde)) / (m - 2))
    )
    HSIC_value /= m * (m - 3)
    return HSIC_value


def cknna(feats_A, feats_B, topk=10, distance_agnostic=False, unbiased=True):
    if feats_A.ndim != 2 or feats_B.ndim != 2:
        raise ValueError(f"Expected 2D feats, got {feats_A.shape} and {feats_B.shape}")
    if feats_A.shape[0] != feats_B.shape[0]:
        raise ValueError(f"N mismatch: feats_A N={feats_A.shape[0]} vs feats_B N={feats_B.shape[0]}")

    n = feats_A.shape[0]

    if topk is None:
        topk = n - 1
    topk = int(topk)
    topk = min(topk, n - (1 if unbiased else 0))
    topk = min(topk, n - 1)
    if topk < 2:
        raise ValueError(f"CKNNA requires topk >= 2, got topk={topk}, n={n}")

    K = feats_A @ feats_A.T
    L = feats_B @ feats_B.T
    device = feats_A.device

    def similarity(Kmat, Lmat, k):
        if unbiased:
            K_hat = Kmat.clone().fill_diagonal_(float("-inf"))
            L_hat = Lmat.clone().fill_diagonal_(float("-inf"))
        else:
            K_hat, L_hat = Kmat, Lmat

        _, topk_K_indices = torch.topk(K_hat, k, dim=1)
        _, topk_L_indices = torch.topk(L_hat, k, dim=1)

        mask_K = torch.zeros(n, n, device=device).scatter_(1, topk_K_indices, 1)
        mask_L = torch.zeros(n, n, device=device).scatter_(1, topk_L_indices, 1)
        mask = mask_K * mask_L

        if distance_agnostic:
            sim = mask * 1.0
        else:
            sim = hsic_unbiased(mask * Kmat, mask * Lmat) if unbiased else torch.tensor(0.0, device=device)
        return sim

    sim_kl = similarity(K, L, topk)
    sim_kk = similarity(K, K, topk)
    sim_ll = similarity(L, L, topk)

    denom = (torch.sqrt(sim_kk * sim_ll) + 1e-6)
    return (sim_kl / denom).item()


def remove_outliers(feats, q, exact=False, max_threshold=None):
    if q == 1:
        return feats

    if exact:
        q_val = feats.view(-1).abs().sort().values[int(q * feats.numel())]
    else:
        q_val = torch.quantile(feats.abs().flatten(start_dim=1), q, dim=1).mean()

    if max_threshold is not None:
        max_threshold = max(max_threshold, q_val)

    return feats.clamp(-q_val, q_val)


def prepare_features(feats, device, q=0.95, exact=False):
    feats = feats.float()
    feats = remove_outliers(feats, q=q, exact=exact)
    feats = feats.to(device)
    feats = F.normalize(feats, dim=-1)
    return feats
