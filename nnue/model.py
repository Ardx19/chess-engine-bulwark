"""N1 / N1-WDL: 768 -> N (shared by both perspectives), SCReLU, concat(us, them) -> head (1 scalar logit or 3 W/D/L logits).

Scalar head: p = sigmoid(out), engine units u = 400 * out.  WDL head: softmax over [W, D, L] logits.
Dual head (head = 4): [expected-score logit, W, D, L]. The scalar output is trained exactly like N1 and is the engine's
search eval; the W/D/L outputs feed the draw-aware utility. With wdl_detach the W/D/L layer is a read-out of the
trunk that sends no gradient into it, so the trunk trains as in N1.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import PAD

WEIGHT_CLAMP = 1.98   # first-layer weights stay inside int16 range after the QA = 255 scaling; output weights stay
                      # small enough (|w * QB| * QA < 2^15) for the engine's exact int16 SIMD v*w kernel


class NNUE(nn.Module):
    def __init__(self, n: int = 256, head: int = 1, wdl_detach: bool = False, wdl_weight: float = 1.0):
        super().__init__()
        assert head in (1, 3, 4)
        self.n, self.head, self.wdl_detach, self.wdl_weight = n, head, bool(wdl_detach), float(wdl_weight)
        self.ft = nn.EmbeddingBag(PAD + 1, n, mode="sum", padding_idx=PAD)
        self.ft_bias = nn.Parameter(torch.zeros(n))
        if head == 4:
            self.out_es = nn.Linear(2 * n, 1)
            self.out = nn.Linear(2 * n, 3)
        else:
            self.out = nn.Linear(2 * n, head)
        nn.init.normal_(self.ft.weight, std=0.05)
        with torch.no_grad():
            self.ft.weight[PAD].zero_()

    @staticmethod
    def screlu(x):
        return torch.clamp(x, 0.0, 1.0) ** 2

    def forward(self, us, them):
        a = self.ft(us) + self.ft_bias
        b = self.ft(them) + self.ft_bias
        x = torch.cat([self.screlu(a), self.screlu(b)], dim=1)
        if self.head == 4:
            return torch.cat([self.out_es(x), self.out(x.detach() if self.wdl_detach else x)], dim=1)
        o = self.out(x)
        return o.squeeze(1) if self.head == 1 else o

    @torch.no_grad()
    def clamp_(self):
        self.ft.weight.clamp_(-WEIGHT_CLAMP, WEIGHT_CLAMP)
        self.ft.weight[PAD].zero_()
        self.out.weight.clamp_(-WEIGHT_CLAMP, WEIGHT_CLAMP)
        if self.head == 4:
            self.out_es.weight.clamp_(-WEIGHT_CLAMP, WEIGHT_CLAMP)


def loss_fn(model: NNUE, out, target, escore, alpha: float = 0.0):
    """Scalar: MSE(sigmoid(out), expected score). WDL: soft cross-entropy + alpha * (expected score error)^2.
    Dual: scalar loss on out[:, 0] + wdl_weight * WDL loss on out[:, 1:4]."""
    aux = getattr(model, "aux_loss", 0.0)
    if model.head == 1:
        return F.mse_loss(torch.sigmoid(out), escore) + aux
    if model.head == 4:
        scalar = F.mse_loss(torch.sigmoid(out[:, 0]), escore)
        logp = F.log_softmax(out[:, 1:4], dim=1)
        wdl = -(target * logp).sum(dim=1).mean()
        if alpha:
            p = logp.exp()
            wdl = wdl + alpha * ((p[:, 0] + 0.5 * p[:, 1] - escore) ** 2).mean()
        return scalar + model.wdl_weight * wdl + aux
    logp = F.log_softmax(out, dim=1)
    ce = -(target * logp).sum(dim=1).mean()
    if alpha:
        p = logp.exp()
        pe = p[:, 0] + 0.5 * p[:, 1]
        ce = ce + alpha * ((pe - escore) ** 2).mean()
    return ce + aux


@torch.no_grad()
def metrics(model: NNUE, out, target, escore):
    """Teacher agreement: log loss (soft cross-entropy for WDL, binary cross-entropy for scalar) and Brier score."""
    if model.head == 1:
        p = torch.sigmoid(out).clamp(1e-6, 1 - 1e-6)
        ll = -(escore * p.log() + (1 - escore) * (1 - p).log()).mean()
        brier = ((p - escore) ** 2).mean()
    else:
        out = out[:, 1:4] if model.head == 4 else out           # dual: W/D/L metrics (comparable with N1-WDL)
        p = F.softmax(out, dim=1)
        ll = -(target * F.log_softmax(out, dim=1)).sum(dim=1).mean()
        brier = ((p - target) ** 2).sum(dim=1).mean()
    return float(ll), float(brier)


# ---------------------------------------------------------------------------------------------------------------
# N2 / N3: king-bucketed L1 (8 mirrored buckets) -> SCReLU -> concat(us, them) -> K expert stacks 2N -> L2 -> L3 -> head.
# K = 1 is N2 (dense). N3 variants: router "count" (hard, by piece count: (pieces - 1) * K // 32) or "learned"
# (top-1 Switch routing on [2N activations, 4 scalars]: y = p_i(x) * E_i(x), plus the load-balancing loss
# alpha * K * sum_i f_i * P_i, stored in `aux_loss` after each forward).
# Engine quantisation (src/nnue.cpp): L1 int16 (QA = 255), activations x_q = round(clamp(acc)^2 / 512) in 0..127,
# L2 and router weights int8 (QB = 64); L3 and the head stay float.

from .data import N2_PAD, N_SCALARS   # noqa: E402

ROUTERS = ("none", "count", "learned")


class NNUE2(nn.Module):
    def __init__(self, n: int = 512, head: int = 3, l2: int = 16, l3: int = 32, k: int = 1, router: str = "none",
                 balance: float = 0.01):
        super().__init__()
        assert router in ROUTERS and (k == 1) == (router == "none")
        self.n, self.head, self.l2n, self.l3n, self.k, self.router, self.balance = n, head, l2, l3, k, router, balance
        self.ft = nn.EmbeddingBag(N2_PAD + 1, n, mode="sum", padding_idx=N2_PAD)
        self.ft_bias = nn.Parameter(torch.zeros(n))
        self.l2 = nn.Linear(2 * n, k * l2)
        self.l3_w = nn.Parameter(torch.empty(k, l3, l2))
        self.l3_b = nn.Parameter(torch.zeros(k, l3))
        self.out_w = nn.Parameter(torch.empty(k, head, l3))
        self.out_b = nn.Parameter(torch.zeros(k, head))
        nn.init.normal_(self.ft.weight, std=0.05)
        nn.init.uniform_(self.l3_w, -1 / l2 ** 0.5, 1 / l2 ** 0.5)
        nn.init.uniform_(self.out_w, -1 / l3 ** 0.5, 1 / l3 ** 0.5)
        if router == "learned":
            self.router_x = nn.Linear(2 * n, k)
            self.router_s = nn.Linear(N_SCALARS, k, bias=False)
        with torch.no_grad():
            self.ft.weight[N2_PAD].zero_()
        self.aux_loss = torch.zeros(())

    def expert_index(self, x, scalars, pcount):
        if self.router == "none":
            return torch.zeros(len(x), dtype=torch.long, device=x.device), None
        if self.router == "count":
            return torch.clamp((pcount - 1) * self.k // 32, 0, self.k - 1), None
        logits = self.router_x(x) + self.router_s(scalars)
        p = F.softmax(logits.float(), dim=1)
        return p.argmax(dim=1), p

    def forward(self, us, them, scalars=None, pcount=None):
        a = self.ft(us) + self.ft_bias
        b = self.ft(them) + self.ft_bias
        x = torch.cat([NNUE.screlu(a), NNUE.screlu(b)], dim=1)
        idx, p = self.expert_index(x, scalars, pcount)
        ar = torch.arange(len(x), device=x.device)
        h2 = self.l2(x).view(len(x), self.k, self.l2n)[ar, idx].clamp(0, 1)
        h3 = (torch.bmm(self.l3_w[idx], h2.unsqueeze(2)).squeeze(2) + self.l3_b[idx]).clamp(0, 1)
        o = torch.bmm(self.out_w[idx], h3.unsqueeze(2)).squeeze(2) + self.out_b[idx]
        if p is not None:
            o = o * p[ar, idx].unsqueeze(1).to(o.dtype)
            f = torch.bincount(idx, minlength=self.k).float() / len(idx)
            self.aux_loss = self.balance * self.k * (f * p.mean(0)).sum()
        else:
            self.aux_loss = torch.zeros((), device=x.device)
        return o.squeeze(1) if self.head == 1 else o

    @torch.no_grad()
    def clamp_(self):
        self.ft.weight.clamp_(-WEIGHT_CLAMP, WEIGHT_CLAMP)
        self.ft.weight[N2_PAD].zero_()
        self.l2.weight.clamp_(-WEIGHT_CLAMP, WEIGHT_CLAMP)
        if self.router == "learned":
            self.router_x.weight.clamp_(-WEIGHT_CLAMP, WEIGHT_CLAMP)

    @torch.no_grad()
    def upcycle_from(self, dense: "NNUE2", noise: float = 0.01):
        """Sparse upcycling: copy the trained N2 (K = 1) into every expert, with a little noise to break symmetry."""
        assert dense.k == 1 and dense.n == self.n and dense.l2n == self.l2n and dense.l3n == self.l3n
        self.ft.weight.copy_(dense.ft.weight)
        self.ft_bias.copy_(dense.ft_bias)
        for e in range(self.k):
            sl = slice(e * self.l2n, (e + 1) * self.l2n)
            self.l2.weight[sl].copy_(dense.l2.weight + noise * torch.randn_like(dense.l2.weight))
            self.l2.bias[sl].copy_(dense.l2.bias)
            self.l3_w[e].copy_(dense.l3_w[0]); self.l3_b[e].copy_(dense.l3_b[0])
            self.out_w[e].copy_(dense.out_w[0]); self.out_b[e].copy_(dense.out_b[0])


# ---------------------------------------------------------------------------------------------------------------
# N4: 8 mirrored king buckets x 768 -> N (shared by both perspectives), SCReLU, concat(us, them) -> 8 output buckets
# selected by piece count ((pieces - 1) * 8 // 32), each a linear head: 1 scalar, 3 W/D/L, or 4 = dual
# [expected-score logit, W, D, L] (wdl_detach / wdl_weight as in N1). No hidden layers, so the engine's quantised
# path is the exact int16 one used by N1 (export format v3).

N_OUT_BUCKETS = 8


class NNUE4(nn.Module):
    def __init__(self, n: int = 1536, head: int = 4, wdl_detach: bool = False, wdl_weight: float = 1.0,
                 out_buckets: int = N_OUT_BUCKETS):
        super().__init__()
        assert head in (1, 3, 4)
        self.n, self.head, self.ob = n, head, out_buckets
        self.wdl_detach, self.wdl_weight = bool(wdl_detach), float(wdl_weight)
        self.ft = nn.EmbeddingBag(N2_PAD + 1, n, mode="sum", padding_idx=N2_PAD)
        self.ft_bias = nn.Parameter(torch.zeros(n))
        if head in (1, 4):
            self.out_es = nn.Linear(2 * n, out_buckets)
        if head in (3, 4):
            self.out = nn.Linear(2 * n, out_buckets * 3)
        nn.init.normal_(self.ft.weight, std=0.05)
        with torch.no_grad():
            self.ft.weight[N2_PAD].zero_()
        self.aux_loss = torch.zeros(())

    def bucket(self, pcount):
        return torch.clamp((pcount - 1) * self.ob // 32, 0, self.ob - 1)

    def forward(self, us, them, scalars=None, pcount=None):
        a = self.ft(us) + self.ft_bias
        b = self.ft(them) + self.ft_bias
        x = torch.cat([NNUE.screlu(a), NNUE.screlu(b)], dim=1)
        ar = torch.arange(len(x), device=x.device)
        k = self.bucket(pcount)
        outs = []
        if self.head in (1, 4):
            outs.append(self.out_es(x)[ar, k].unsqueeze(1))
        if self.head in (3, 4):
            xw = x.detach() if (self.head == 4 and self.wdl_detach) else x
            outs.append(self.out(xw).view(len(x), self.ob, 3)[ar, k])
        o = torch.cat(outs, dim=1)
        return o.squeeze(1) if self.head == 1 else o

    @torch.no_grad()
    def clamp_(self):
        self.ft.weight.clamp_(-WEIGHT_CLAMP, WEIGHT_CLAMP)
        self.ft.weight[N2_PAD].zero_()
        for layer in ("out_es", "out"):
            if hasattr(self, layer):
                getattr(self, layer).weight.clamp_(-WEIGHT_CLAMP, WEIGHT_CLAMP)


def build_model(cfg: dict) -> nn.Module:
    arch = cfg.get("arch", "n1")
    if arch == "n1":
        return NNUE(cfg["n"], cfg["head"], bool(cfg.get("wdl_detach") or 0), cfg.get("wdl_weight") or 1.0)
    if arch == "n4":
        return NNUE4(cfg["n"], cfg["head"], bool(cfg.get("wdl_detach") or 0), cfg.get("wdl_weight") or 1.0)
    return NNUE2(cfg["n"], cfg["head"], cfg.get("l2", 16), cfg.get("l3", 32), cfg.get("k", 1),
                 cfg.get("router", "none"), cfg.get("balance", 0.01))


def run_model(model: nn.Module, batch: dict, device):
    """Forward a loader batch (numpy) through either architecture."""
    # .long(): multi-process loaders ship int16 indices
    us, them = torch.from_numpy(batch["us"]).to(device).long(), torch.from_numpy(batch["them"]).to(device).long()
    if isinstance(model, (NNUE2, NNUE4)):
        return model(us, them, torch.from_numpy(batch["scalars"]).to(device), torch.from_numpy(batch["pcount"]).to(device))
    return model(us, them)
