import torch
from torch_geometric.loader import DataLoader
from sklearn.metrics import r2_score
from tqdm import tqdm

def train_epoch(model, loader, optimizer, loss_fn):
    model.train()
    total_loss = 0
    for batch in loader:
        optimizer.zero_grad()
        out = model(batch.x, batch.edge_index, batch.batch)
        loss = loss_fn(out, batch.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def evaluate(model, loader):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for batch in loader:
            out = model(batch.x, batch.edge_index, batch.batch)
            y_true += batch.y.flatten().tolist()
            y_pred += out.flatten().tolist()
    return r2_score(y_true, y_pred)

def mc_dropout_predict(model, loader, n_samples=10):
    model.train()  # keep dropout active
    preds = []
    for _ in range(n_samples):
        y_sample = []
        for batch in loader:
            with torch.no_grad():
                out = model(batch.x, batch.edge_index, batch.batch)
                y_sample += out.flatten().tolist()
        preds.append(y_sample)
    return preds
