import os
import glob
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

T = 30      # ventana temporal (frames)
D = 34      # features por frame (COCO17 x,y => 17*2)

class PoseSeqDataset(Dataset):
    def __init__(self, folder):
        self.samples = []
        files = glob.glob(os.path.join(folder, "*.npy"))

        for f in files:
            seq = np.load(f)  # (L,34)
            if seq.shape[0] < T:
                continue

            # cortes en ventanas de T (no solapadas para MVP)
            for i in range(0, seq.shape[0] - T + 1, T):
                x = seq[i:i+T]
                self.samples.append(x.astype(np.float32))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

class LSTMAE(nn.Module):
    def __init__(self, d_in=D, hidden=128):
        super().__init__()
        self.encoder = nn.LSTM(d_in, hidden, batch_first=True)
        self.decoder = nn.LSTM(hidden, hidden, batch_first=True)
        self.out = nn.Linear(hidden, d_in)

    def forward(self, x):
        # x: (B,T,D)
        _, (h, _) = self.encoder(x)  # h: (1,B,H)
        h_rep = h[-1].unsqueeze(1).repeat(1, x.size(1), 1)  # (B,T,H)
        y, _ = self.decoder(h_rep)
        y = self.out(y)
        return y

def main():
    train_folder = "data/poses/normal"
    out_model = "models/anomaly/pose_ae.pt"
    os.makedirs(os.path.dirname(out_model), exist_ok=True)

    ds = PoseSeqDataset(train_folder)
    if len(ds) == 0:
        raise RuntimeError("No hay muestras para entrenar. ¿Generaste npy en data/poses/normal?")

    dl = DataLoader(ds, batch_size=64, shuffle=True, drop_last=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = LSTMAE().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    print(f"Train samples: {len(ds)} | device: {device}")

    epochs = 10
    for epoch in range(epochs):
        model.train()
        total = 0.0
        for x in dl:
            x = x.to(device)
            xhat = model(x)
            loss = loss_fn(xhat, x)

            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()

        print(f"Epoch {epoch+1}/{epochs} - loss: {total/len(dl):.6f}")

    torch.save(model.state_dict(), out_model)
    print(f"✅ Guardado: {out_model}")

if __name__ == "__main__":
    main()
