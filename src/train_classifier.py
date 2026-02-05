import os
import glob
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

WIN = 32
D = 34

DATA_ROOT = "data/windows_dataset"  # <-- salida del make_windows
NORMAL_DIR = os.path.join(DATA_ROOT, "normal")
SUSPECT_DIR = os.path.join(DATA_ROOT, "suspect")

class WindowClsDataset(Dataset):
    def __init__(self, normal_dir, suspect_dir):
        self.samples = []

        for f in glob.glob(os.path.join(normal_dir, "*.npy")):
            x = np.load(f)  # (WIN,34)
            if x.shape != (WIN, D):
                continue
            self.samples.append((x.astype(np.float32), 0))  # 0=normal

        for f in glob.glob(os.path.join(suspect_dir, "*.npy")):
            x = np.load(f)  # (WIN,34)
            if x.shape != (WIN, D):
                continue
            self.samples.append((x.astype(np.float32), 1))  # 1=suspect

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y = self.samples[idx]
        return torch.tensor(x), torch.tensor(y, dtype=torch.long)

class TCNClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(D, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        self.fc = nn.Linear(128, 2)

    def forward(self, x):
        # x: (B,WIN,D)
        x = x.permute(0, 2, 1)  # (B,D,WIN)
        x = self.net(x).squeeze(-1)  # (B,128)
        return self.fc(x)

def main():
    out_model = "models/classifier/pose_cls.pt"
    os.makedirs(os.path.dirname(out_model), exist_ok=True)

    ds = WindowClsDataset(NORMAL_DIR, SUSPECT_DIR)
    if len(ds) == 0:
        raise RuntimeError("No hay ventanas. ¿Corriste make_windows_with_ranges.py?")

    # split train/val
    val_ratio = 0.15
    n_val = int(len(ds) * val_ratio)
    n_train = len(ds) - n_val
    train_ds, val_ds = random_split(ds, [n_train, n_val])

    dl = DataLoader(train_ds, batch_size=256, shuffle=True, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=256, shuffle=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TCNClassifier().to(device)

    # class weights (por si hay desbalance)
    ys = [y for _, y in ds.samples]
    n0 = sum(1 for y in ys if y == 0)
    n1 = sum(1 for y in ys if y == 1)
    w0 = (n0 + n1) / (2 * max(1, n0))
    w1 = (n0 + n1) / (2 * max(1, n1))
    class_weights = torch.tensor([w0, w1], dtype=torch.float32, device=device)

    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    print(f"Train samples: {len(train_ds)} | Val samples: {len(val_ds)} | device: {device}")
    print(f"Class counts -> normal: {n0} suspect: {n1} | weights: {w0:.3f}, {w1:.3f}")

    epochs = 12
    best_val = 1e9

    for epoch in range(epochs):
        model.train()
        total = 0.0
        for x, y in dl:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = loss_fn(logits, y)

            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()

        # val
        model.eval()
        vloss = 0.0
        correct = 0
        total_n = 0
        with torch.no_grad():
            for x, y in val_dl:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                loss = loss_fn(logits, y)
                vloss += loss.item()

                pred = logits.argmax(dim=1)
                correct += (pred == y).sum().item()
                total_n += y.numel()

        avg_train = total / max(1, len(dl))
        avg_val = vloss / max(1, len(val_dl))
        acc = correct / max(1, total_n)

        print(f"Epoch {epoch+1}/{epochs} - train_loss: {avg_train:.6f} - val_loss: {avg_val:.6f} - val_acc: {acc:.4f}")

        if avg_val < best_val:
            best_val = avg_val
            torch.save(model.state_dict(), out_model)

    print(f"✅ Guardado mejor modelo: {out_model}")

if __name__ == "__main__":
    main()
