"""N1 / N1-WDL: 768 -> N (shared by both perspectives), SCReLU, concat(us, them) -> head (1 scalar logit or 3 W/D/L logits).

Scalar head: p = sigmoid(out), engine units u = 400 * out.  WDL head: softmax over [W, D, L] logits.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import PAD

WEIGHT_CLAMP = 1.98   # first-layer weights stay inside int16 range after the QA = 255 scaling


class NNUE(nn.Module):
    def __init__(self, n: int = 256, head: int = 1):
        super().__init__()
        self.n, self.head = n, head
        self.ft = nn.EmbeddingBag(PAD + 1, n, mode="sum", padding_idx=PAD)
        self.ft_bias = nn.Parameter(torch.zeros(n))
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
        o = self.out(x)
        return o.squeeze(1) if self.head == 1 else o

    @torch.no_grad()
    def clamp_(self):
        self.ft.weight.clamp_(-WEIGHT_CLAMP, WEIGHT_CLAMP)
        self.ft.weight[PAD].zero_()


def loss_fn(model: NNUE, out, target, escore, alpha: float = 0.0):
    """Scalar: MSE(sigmoid(out), expected score). WDL: soft cross-entropy + alpha * (expected score error)^2."""
    if model.head == 1:
        return F.mse_loss(torch.sigmoid(out), escore)
    logp = F.log_softmax(out, dim=1)
    ce = -(target * logp).sum(dim=1).mean()
    if alpha:
        p = logp.exp()
        pe = p[:, 0] + 0.5 * p[:, 1]
        ce = ce + alpha * ((pe - escore) ** 2).mean()
    return ce


@torch.no_grad()
def metrics(model: NNUE, out, target, escore):
    """Teacher agreement: log loss (soft cross-entropy for WDL, binary cross-entropy for scalar) and Brier score."""
    if model.head == 1:
        p = torch.sigmoid(out).clamp(1e-6, 1 - 1e-6)
        ll = -(escore * p.log() + (1 - escore) * (1 - p).log()).mean()
        brier = ((p - escore) ** 2).mean()
    else:
        p = F.softmax(out, dim=1)
        ll = -(target * F.log_softmax(out, dim=1)).sum(dim=1).mean()
        brier = ((p - target) ** 2).sum(dim=1).mean()
    return float(ll), float(brier)
