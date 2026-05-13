# Graph Attention v2

Graph Attention v2 implementation by Shaked Brody and Uri Alon and Eran Yahav.

## Installation

```bash
pip install gatv2
```

## Usage

```py
from gatv2 import GraphAttention

model = GraphAttention(
    in_features = cora.in_features
    hidden_dimension = 8
    num_classes = 10
    heads = 8
    dropout = 0.6
    share_weights = True,
    activation = torch.nn.ELU
    layer_activation = torch.nn.LeakyReLU
)

print(model)
```

## Citation

```bibtex
@misc{2105.14491,
    Title = {How Attentive are Graph Attention Networks?},
    Author = {Shaked Brody and Uri Alon and Eran Yahav},
    Year = {2021},
    Eprint = {arXiv:2105.14491},
}
```
