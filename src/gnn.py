"""Plain-torch message-passing GNN decoder over fired detectors."""
import torch
from torch import nn

from src.graph import GraphBatch

EMPTY_LOGIT = -20.0


def _mlp(i: int, h: int, o: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(i, h), nn.SiLU(), nn.Linear(h, o))


class _Block(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.msg = _mlp(3 * hidden, hidden, hidden)
        self.upd = _mlp(2 * hidden, hidden, hidden)
        self.norm = nn.LayerNorm(hidden)

    def forward(self, h, e, edge_index):
        src, dst = edge_index
        m = self.msg(torch.cat([h[src], h[dst], e], dim=1))
        agg = torch.zeros_like(h).index_add_(0, dst, m)
        return self.norm(h + self.upd(torch.cat([h, agg], dim=1)))


class GNNDecoder(nn.Module):
    def __init__(self, node_dim: int = 8, edge_dim: int = 4, hidden: int = 64, layers: int = 4):
        super().__init__()
        self.node_enc = _mlp(node_dim, hidden, hidden)
        self.edge_enc = _mlp(edge_dim, hidden, hidden)
        self.blocks = nn.ModuleList(_Block(hidden) for _ in range(layers))
        self.head = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.SiLU(), nn.Linear(hidden, 1))

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(self, g: GraphBatch) -> torch.Tensor:
        logits = torch.full((g.n_graphs,), EMPTY_LOGIT, device=g.x.device)
        if g.x.shape[0] == 0:
            return logits
        h = self.node_enc(g.x)
        e = self.edge_enc(g.edge_attr)
        for blk in self.blocks:
            h = blk(h, e, g.edge_index)
        idx = g.batch[:, None].expand_as(h)
        shape = (g.n_graphs, h.shape[1])
        mean = torch.zeros(shape, device=h.device).scatter_reduce(0, idx, h, "mean", include_self=False)
        mx = torch.zeros(shape, device=h.device).scatter_reduce(0, idx, h, "amax", include_self=False)
        out = self.head(torch.cat([mean, mx], dim=1)).squeeze(1)
        return torch.where(g.nonempty, out, logits)
