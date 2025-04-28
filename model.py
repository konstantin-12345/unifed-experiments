# File: model.py

import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------
# Image classification model
# ----------------------------
class CNN(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),  # assuming 3-channel input
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 32×16×16
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 64×8×8
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1,1)),  # 128×1×1
        )
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# ----------------------------
# Text classification model
# ----------------------------
class TextRNN(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 128,
        hidden_dim: int = 256,
        num_layers: int = 2,
        num_classes: int = 4,
        pad_idx: int = 0,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.3,
        )
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x: torch.Tensor, lengths=None) -> torch.Tensor:
        """
        x: LongTensor(batch_size, seq_len)
        lengths: optional tensor of actual sequence lengths
        """
        embed = self.embedding(x)  # (batch, seq_len, embed_dim)
        packed = nn.utils.rnn.pack_padded_sequence(embed, lengths, batch_first=True, enforce_sorted=False) \
                 if lengths is not None else embed
        out, _ = self.lstm(packed)
        if lengths is not None:
            out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
            # take last valid output for each sequence
            idx = (lengths - 1).view(-1,1).expand(len(lengths), out.size(2)).unsqueeze(1)
            out = out.gather(1, idx).squeeze(1)
        else:
            # use last time-step output
            out = out[:, -1, :]
        logits = self.fc(out)
        return logits

# ----------------------------
# Tabular / structured data model
# ----------------------------
class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list = [64, 32], output_dim: int = 2):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU(inplace=True)]
            prev = h
        layers.append(nn.Linear(prev, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

# ----------------------------
# Factory to create model by dataset
# ----------------------------
def create_model(dataset_name: str, **kwargs) -> nn.Module:
    """
    Returns a model instance appropriate for the given dataset.
    For structured datasets, you must pass `input_dim` and optionally `output_dim`.
    For text datasets, you must pass `vocab_size` and optionally other RNN params.
    """
    if dataset_name in ["cifar10", "cifar100", "femnist"]:
        num_classes = {"cifar10": 10, "cifar100": 100, "femnist": 62}[dataset_name]
        return CNN(num_classes=num_classes)

    elif dataset_name in ["shakespeare", "ag_news"]:
        vocab_size = kwargs.get("vocab_size")
        if vocab_size is None:
            raise ValueError("For text datasets, please pass vocab_size")
        embed_dim   = kwargs.get("embed_dim", 128)
        hidden_dim  = kwargs.get("hidden_dim", 256)
        num_layers  = kwargs.get("num_layers", 2)
        num_classes = kwargs.get("num_classes", 4)
        pad_idx     = kwargs.get("pad_idx", 0)
        return TextRNN(vocab_size, embed_dim, hidden_dim, num_layers, num_classes, pad_idx)

    elif dataset_name in ["adult", "diabetes"]:
        input_dim  = kwargs.get("input_dim")
        if input_dim is None:
            raise ValueError("For structured datasets, please pass input_dim")
        hidden_dims = kwargs.get("hidden_dims", [64, 32])
        output_dim  = kwargs.get("output_dim", 2)
        return MLP(input_dim, hidden_dims, output_dim)

    else:
        raise NotImplementedError(f"Model for dataset '{dataset_name}' is not implemented.")
