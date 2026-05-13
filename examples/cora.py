import torch
from gatv2 import GraphAttention
from types import SimpleNamespace
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import to_dense_adj


def main(
    epochs: int = 200,
    lr: float = 0.005,
    weight_decay: float = 0.0005,
    heads: int = 8,
    dropout: float = 0.6,
):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    dataset = load_cora(device)
    model = GraphAttention(
        in_features=dataset.num_features,
        num_classes=dataset.num_classes,
        hidden_dimension=8,
        heads=heads,
        dropout=dropout,
    )
    model = model.to(device)
    optimizer = get_optimizer(model=model, lr=lr, weight_decay=weight_decay)
    criterion = get_criterion()

    for epoch in range(epochs):
        train_step(
            model=model,
            optimizer=optimizer,
            x=dataset.x,
            y=dataset.y,
            matrix=dataset.matrix,
            criterion=criterion,
            train_mask=dataset.mask.train,
        )

        if epoch % 10 != 0:
            continue

        validation_stats = validate(
            model=model,
            matrix=dataset.matrix,
            x=dataset.x,
            y=dataset.y,
            validation_mask=dataset.mask.validation,
            criterion=criterion,
        )
        print(
            f"Validation Accuracy: {validation_stats.accuracy} | Validation Loss: {validation_stats.loss}"
        )


def load_cora(device):
    dataset = Planetoid(root="./data", name="Cora")
    cora_data = dataset[0]
    x = cora_data.x.to(device)
    y = cora_data.y.to(device)
    matrix = to_dense_adj(cora_data.edge_index, max_num_nodes=cora_data.num_nodes)
    matrix = matrix[0].unsqueeze(-1).to(device)
    return SimpleNamespace(
        **{
            "x": x,
            "y": y,
            "matrix": matrix,
            "num_features": dataset.num_features,
            "num_classes": dataset.num_classes,
            "mask": SimpleNamespace(
                **{
                    "validation": cora_data.val_mask,
                    "train": cora_data.train_mask,
                }
            ),
        }
    )


def get_optimizer(*, model, lr: float, weight_decay: float):
    return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)


def get_criterion():
    return torch.nn.CrossEntropyLoss()


def train_step(
    *,
    model,
    optimizer,
    x,
    y,
    matrix,
    criterion,
    train_mask,
):
    model.train()
    optimizer.zero_grad()
    logits = model(x, matrix)
    loss = criterion(logits[train_mask], y[train_mask])
    loss.backward()
    optimizer.step()


@torch.inference_mode()
def validate(model, matrix, x, y, criterion, validation_mask):
    model.eval()
    validation_logits = model(x, matrix)
    validation_loss = criterion(validation_logits[validation_mask], y[validation_mask])
    predictions = validation_logits[validation_mask].argmax(dim=1)
    correct = (predictions == y[validation_mask]).sum().item()
    accuracy = correct / validation_mask.sum().item()
    return SimpleNamespace(**{"accuracy": accuracy, "loss": validation_loss})


if __name__ == "__main__":
    main()
