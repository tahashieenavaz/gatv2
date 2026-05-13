import torch
from typing import Type


class GraphAttentionLayer(torch.nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        heads: int,
        is_concat: bool,
        dropout: float,
        share_weights: bool,
        activation: Type[torch.nn.Module],
    ):
        super().__init__()
        self.is_concat = is_concat
        self.heads = heads
        self.n_hidden = out_features // heads if is_concat else out_features

        self.alpha = torch.nn.Linear(in_features, self.n_hidden * heads, bias=False)
        self.beta = (
            self.alpha
            if share_weights
            else torch.nn.Linear(in_features, self.n_hidden * heads, bias=False)
        )

        self.attention = torch.nn.Linear(self.n_hidden, 1, bias=False)
        self.activation = activation()
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, h: torch.Tensor, matrix: torch.Tensor):
        n_nodes = h.size(0)

        # project and reshape: (N, H, F)
        g_l = self.alpha(h).view(n_nodes, self.heads, self.n_hidden)
        g_r = self.beta(h).view(n_nodes, self.heads, self.n_hidden)

        # broadcast to create N x N pairs: (N, 1, H, F) + (1, N, H, F) -> (N, N, H, F)
        g_sum = g_l.unsqueeze(1) + g_r.unsqueeze(0)

        # compute attention scores: (N, N, H)
        e = self.attention(self.activation(g_sum)).squeeze(-1)

        # mask disconnected nodes and apply softmax
        e = e.masked_fill(matrix == 0, float("-inf"))

        attention = torch.softmax(e, dim=1)
        attention = self.dropout(attention)

        # apply attention to right node features: (N, H, F)
        attention_residual = torch.einsum("ijh,jhf->ihf", attention, g_r)

        return (
            attention_residual.reshape(n_nodes, -1)
            if self.is_concat
            else attention_residual.mean(dim=1)
        )
