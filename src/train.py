import copy
import os

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split

from src.build_dataset import build_dataset
from src.model import SeizureCNN


def evaluate(model, X, y, threshold=0.5):
    """Run the model on (X, y) and return confusion-matrix metrics (positive = seizure)."""
    model.eval()
    with torch.no_grad():
        probs = torch.sigmoid(model(X))
        preds = (probs > threshold).float()

    tp = ((preds == 1) & (y == 1)).sum().item()
    fp = ((preds == 1) & (y == 0)).sum().item()
    tn = ((preds == 0) & (y == 0)).sum().item()
    fn = ((preds == 0) & (y == 1)).sum().item()

    # Guard against division by zero when a class is absent from predictions.
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / (tp + fp + tn + fn)

    return {
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "precision": precision, "recall": recall, "f1": f1, "accuracy": accuracy,
    }


def main():
    # 1. Build dataset
    X, y = build_dataset(
        data_dir="data/raw/chb01",
        summary_path="data/raw/chb01/chb01-summary.txt",
        filenames=[
            "chb01_01.edf", "chb01_02.edf", "chb01_03.edf", "chb01_04.edf",
            "chb01_05.edf", "chb01_06.edf", "chb01_07.edf", "chb01_15.edf",
            "chb01_16.edf", "chb01_18.edf", "chb01_21.edf", "chb01_26.edf"
        ]
    )

    # 2. Three-way stratified split: hold out a test set first, then carve a
    # validation set off the remainder. Stratify keeps the seizure ratio
    # similar across all three splits. Final proportions: 64% / 16% / 20%.
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.2, random_state=42, stratify=y_temp
    )

    print(f"Train: {X_train.shape[0]} samples, {y_train.sum()} seizure")
    print(f"Val:   {X_val.shape[0]} samples, {y_val.sum()} seizure")
    print(f"Test:  {X_test.shape[0]} samples, {y_test.sum()} seizure")

    # Normalize using TRAINING statistics only (avoid leakage into val/test).
    mean = X_train.mean()
    std = X_train.std()
    X_train = (X_train - mean) / std
    X_val = (X_val - mean) / std
    X_test = (X_test - mean) / std
    print(f"Normalized — train mean: {X_train.mean():.4f}, std: {X_train.std():.4f}")

    # 3. Convert to PyTorch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)  # shape (N, 1)
    X_val = torch.tensor(X_val, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

    # 4. Set up device (use Apple GPU if available)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print("Using device:", device)

    model = SeizureCNN().to(device)
    # Val and test sets are small enough to evaluate in one pass — keep them on the device.
    X_val, y_val = X_val.to(device), y_val.to(device)
    X_test, y_test = X_test.to(device), y_test.to(device)

    # 5. Handle class imbalance with a class-balanced sampler.
    # With ~1% positives, plain random mini-batches mostly contain zero seizure
    # windows, which collapses training. Weight each sample by the inverse of
    # its class frequency so every batch is, on average, ~50/50 seizure/normal.
    num_normal = (y_train == 0).sum().item()
    num_seizure = (y_train == 1).sum().item()
    print(f"Class balance — normal: {num_normal}, seizure: {num_seizure}")

    labels = y_train.long().squeeze(1)  # (N,)
    class_weights = torch.tensor([1.0 / num_normal, 1.0 / num_seizure])
    sample_weights = class_weights[labels]
    sampler = WeightedRandomSampler(
        sample_weights, num_samples=len(sample_weights), replacement=True
    )

    # Tensors stay on CPU here; each batch is moved to the device in the loop.
    train_loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=64,
        sampler=sampler,  # sampler handles shuffling + balancing
    )

    # Balanced batches already correct the imbalance, so plain BCE (no
    # pos_weight) — adding pos_weight on top would over-bias toward positives.
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # 6. Training loop with early stopping on validation F1.
    # The balanced sampler oversamples the 74 seizure windows with replacement,
    # so the model overfits quickly — we keep the weights from the best val-F1
    # epoch and stop once it hasn't improved for `patience` epochs.
    n_epochs = 100
    patience = 15
    best_f1 = -1.0
    best_state = None
    best_epoch = 0
    epochs_no_improve = 0

    for epoch in range(n_epochs):
        model.train()
        running_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)

            optimizer.zero_grad()
            outputs = model(xb)
            loss = criterion(outputs, yb)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * xb.shape[0]  # sum, weighted by batch size

        avg_loss = running_loss / len(train_loader.dataset)
        val = evaluate(model, X_val, y_val)

        if val["f1"] > best_f1:
            best_f1 = val["f1"]
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch + 1
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{n_epochs}, Loss: {avg_loss:.4f}, "
                  f"Val F1: {val['f1']:.3f} (P {val['precision']:.2f} / R {val['recall']:.2f})")

        if epochs_no_improve >= patience:
            print(f"Early stopping at epoch {epoch+1} — no val F1 improvement for {patience} epochs.")
            break

    # Restore the best-performing weights before the final test evaluation.
    if best_state is not None:
        model.load_state_dict(best_state)
    print(f"\nBest model: epoch {best_epoch}, Val F1: {best_f1:.3f}")

    # Persist the best checkpoint. We store the normalization statistics
    # alongside the weights because any future inference must apply the exact
    # same (mean, std) used at training time.
    os.makedirs("models", exist_ok=True)
    checkpoint_path = "models/seizure_cnn.pt"
    torch.save(
        {
            "model_state": best_state,
            "norm_mean": float(mean),
            "norm_std": float(std),
            "best_epoch": best_epoch,
            "best_val_f1": best_f1,
        },
        checkpoint_path,
    )
    print(f"Saved checkpoint to {checkpoint_path}")

    # 7. Final evaluation on the held-out test set (best model).
    m = evaluate(model, X_test, y_test)
    print(f"\nTest accuracy: {m['tp'] + m['tn']}/{X_test.shape[0]} ({100*m['accuracy']:.1f}%)")
    print(f"Confusion matrix:  TP={m['tp']}  FP={m['fp']}  TN={m['tn']}  FN={m['fn']}")
    print(f"Seizures caught (recall): {m['tp']}/{m['tp'] + m['fn']} ({100*m['recall']:.1f}%)")
    print(f"Precision: {100*m['precision']:.1f}%  (of {m['tp'] + m['fp']} seizure alarms, {m['tp']} were real)")
    print(f"F1 score:  {m['f1']:.3f}")


if __name__ == "__main__":
    main()