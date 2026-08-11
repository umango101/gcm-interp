import torch
from tqdm import tqdm
import sys
import gc
import sys
import einops
from patching_utils import PatchingUtils
sys.path.insert(1, '../atp/')
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split

class HeadProbe(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.linear = nn.Linear(hidden_dim, 1, bias=False)
    
    def forward(self, x):
        return torch.sigmoid(self.linear(x)).squeeze(-1)


class LinearProbes:
    def __init__(self, model_handler, batch_handler, config):
        self.model_handler = model_handler
        self.batch_handler = batch_handler
        self.data_handler = batch_handler.data_handler
        self.config = config
        self.batch_size = self.config.args.batch_size
        self.patching_utils = PatchingUtils(self)
        self.get_reps = self.patching_utils.get_activations
        self.print_now = False

    def train_probe(self, probe, train_loader, val_loader, device, epochs=10, lr=1e-3):
        probe = probe.to(device)
        criterion = nn.BCELoss()
        optimizer = optim.Adam(probe.parameters(), lr=lr)

        for epoch in range(epochs):
            probe.train()
            total_loss = 0
            for x_batch, y_batch in train_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                optimizer.zero_grad()
                preds = probe(x_batch)
                loss = criterion(preds, y_batch)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            avg_loss = total_loss / len(train_loader)
            # print(f"Epoch {epoch+1}, Training Loss: {avg_loss:.4f}")

        # === Evaluation ===
        probe.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                preds = probe(x_batch) > 0.5
                correct += (preds == y_batch).sum().item()
                total += y_batch.size(0)
        
        acc = correct / total
        # print(f"Validation Accuracy: {acc:.4f}")
        return acc

    def prepare_dataloaders(self, A, B, batch_size=64, split_ratio=0.8):
        X = torch.cat([A, B], dim=0)
        y = torch.cat([
            torch.zeros(len(A), dtype=torch.float32),
            torch.ones(len(B), dtype=torch.float32)
        ])

        dataset = TensorDataset(X, y)
        train_size = int(split_ratio * len(dataset))
        val_size = len(dataset) - train_size
        train_set, val_set = random_split(dataset, [train_size, val_size])

        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=batch_size)

        return train_loader, val_loader
    
    def get_accuracy(self):
        model = self.model_handler.model
        accuracy_dict = {}
        desired_reps = []
        undesired_reps = []
        head_dim = self.model_handler.head_dim  # true head_dim, not hidden_size//n_heads
        for idx in tqdm(range(self.config.args.batch_start, self.data_handler.LEN, self.batch_size)):
            start = idx
            stop = min(idx + self.batch_size, self.data_handler.LEN)
            print(f'Running probes from {start} to {stop}')
            self.batch_handler.update(start, stop)
            base_toks = self.batch_handler.base_toks
            d = self.get_reps(base_toks['desired'], logit=False, align=False)
            u = self.get_reps(base_toks['undesired'], logit=False, align=False)
            d = [d[i].detach().cpu() for i in range(len(d))]
            u = [u[i].detach().cpu() for i in range(len(u))]
            desired_reps.append(torch.stack(d))
            undesired_reps.append(torch.stack(u))
            del d, u
        desired_reps = torch.stack(desired_reps).squeeze(2)
        print('DESIRED REPS SHAPE', desired_reps.shape)
        desired_reps = einops.rearrange(desired_reps, 'b l seq (m head_dim) -> b l seq m head_dim', head_dim=head_dim, m=self.model_handler.num_heads)
        undesired_reps = torch.stack(undesired_reps).squeeze(2)
        undesired_reps = einops.rearrange(undesired_reps, 'b l seq (m head_dim) -> b l seq m head_dim', head_dim=head_dim, m=self.model_handler.num_heads)
        print(desired_reps.shape, undesired_reps.shape)
        for l in tqdm(range(len(model.model.layers)), desc='Layers'):
            accuracy_dict[l] = {}    
            for h in range(self.model_handler.num_heads):
                train_loader, val_loader = self.prepare_dataloaders(desired_reps[:, l, -1, h, :].to(torch.float), undesired_reps[:, l, -1, h, :].to(torch.float), batch_size=self.batch_size)
                accuracy_dict[l][h] = self.train_probe(
                    HeadProbe(self.model_handler.dim), 
                    train_loader, 
                    val_loader, 
                    self.config.args.device, 
                    epochs=10, 
                    lr=1e-3
                )
        return accuracy_dict
