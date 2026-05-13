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
        self.alpha = GraphAttentionLayer(
            in_features,
            hidden_dimension,
            heads,
            is_concat=True,
            dropout=dropout,
            share_weights=share_weights,
            activation=layer_activation,
        )
        self.beta = GraphAttentionLayer(
            hidden_dimension,
            num_classes,
            1,
            is_concat=False,
            dropout=dropout,
            share_weights=share_weights,
            activation=layer_activation,
        )
        self.activation = activation()

    def forward(self, x: torch.Tensor, matrix: torch.Tensor):
        x = self.dropout(x)
        x = self.alpha(x, matrix)
        x = self.activation(x)
        x = self.dropout(x)
        return self.beta(x, matrix)
