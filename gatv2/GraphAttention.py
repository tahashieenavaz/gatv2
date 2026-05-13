import torch
from typing import Type
from .GraphAttentionLayer import GraphAttentionLayer


class GraphAttention(torch.nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_dimension: int,
        num_classes: int,
        heads: int,
        dropout: float = 0.6,
        share_weights: bool = True,
        activation: Type[torch.nn.Module] = torch.nn.ELU,
        layer_activation: Type[torch.nn.Module] = torch.nn.LeakyReLU,
    ):
        super().__init__()
        self.dropout = torch.nn.Dropout(dropout)
        self.layer1 = GraphAttentionLayer(
            in_features,
            hidden_dimension,
            heads,
            is_concat=True,
            dropout=dropout,
            share_weights=share_weights,
            activation=layer_activation,
        )
        self.output = GraphAttentionLayer(
            hidden_dimension,
            num_classes,
            1,
            is_concat=False,
            dropout=dropout,
            share_weights=share_weights,
            activation=layer_activation,
        )
        self.activation = activation()

    def forward(self, x: torch.Tensor, adj_mat: torch.Tensor):
        x = self.dropout(x)
        x = torch.nn.functional.elu(self.layer1(x, adj_mat))
        x = self.dropout(x)
        return self.output(x, adj_mat)
