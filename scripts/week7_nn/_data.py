"""Shared loading / encoding of the week-4 train/test split for the neural-network scripts."""

import pickle

import _bootstrap
import numpy as np
import torch


def load_split():
    with open(_bootstrap.RESULTS_DIR / "csp_and_split.pkl", "rb") as f:
        d = pickle.load(f)
    X_train, X_test, y_train, y_test = d["X_train"], d["X_test"], d["y_train"], d["y_test"]

    classes = sorted(set(y_train))
    label_to_idx = {c: i for i, c in enumerate(classes)}
    y_train_idx = np.array([label_to_idx[v] for v in y_train])
    y_test_idx = np.array([label_to_idx[v] for v in y_test])

    mean, std = X_train.mean(), X_train.std()  # z-score with train statistics only
    to_t = lambda a: torch.tensor((a - mean) / std, dtype=torch.float32)  # noqa: E731
    return {
        "X_train": to_t(X_train), "y_train": torch.tensor(y_train_idx, dtype=torch.long),
        "X_test": to_t(X_test), "y_test": torch.tensor(y_test_idx, dtype=torch.long),
        "mean": mean, "std": std, "classes": classes, "n_times": X_train.shape[2],
    }


def train(model, data, n_epochs=100, batch_size=16, lr=1e-3, weight_decay=1e-4):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = torch.nn.CrossEntropyLoss()
    n_train = len(data["X_train"])
    for epoch in range(n_epochs):
        model.train()
        perm = torch.randperm(n_train)
        total_loss = 0.0
        for i in range(0, n_train, batch_size):
            idx = perm[i:i + batch_size]
            optimizer.zero_grad()
            loss = criterion(model(data["X_train"][idx]), data["y_train"][idx])
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(idx)
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1:3d}: loss={total_loss / n_train:.4f}  test_acc={evaluate(model, data):.4f}")
    return model


def evaluate(model, data):
    model.eval()
    with torch.no_grad():
        return (model(data["X_test"]).argmax(1) == data["y_test"]).float().mean().item()
