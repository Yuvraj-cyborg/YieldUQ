import torch
from tqdm import tqdm

def train_epoch(model, loader, optimizer, loss_fn):
    model.train()
    total_loss = 0
    for batch in tqdm(loader, desc="Training"):
        optimizer.zero_grad()

        out = model(
            batch.x,
            batch.edge_index,
            batch.batch,
            batch.ligand_idx,
            batch.additive_idx,
            batch.base_idx,
            batch.aryl_idx,
        )

        loss = loss_fn(out.view(-1), batch.y.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    y_true, y_pred = [], []
    for batch in tqdm(loader, desc="Evaluating"):
        out = model(
            batch.x,
            batch.edge_index,
            batch.batch,
            batch.ligand_idx,
            batch.additive_idx,
            batch.base_idx,
            batch.aryl_idx,
        )
        y_true += batch.y.view(-1).tolist()
        y_pred += out.view(-1).tolist()

    # Compute R²
    y_true_t = torch.tensor(y_true)
    y_pred_t = torch.tensor(y_pred)
    ss_res = torch.sum((y_true_t - y_pred_t) ** 2)
    ss_tot = torch.sum((y_true_t - torch.mean(y_true_t)) ** 2)
    r2 = 1 - ss_res / ss_tot
    return r2.item()


@torch.no_grad()
def mc_dropout_predict(model, loader, n_samples=20):
    model.train()  # enable dropout
    preds = []

    for _ in tqdm(range(n_samples), desc="MC Sampling"):
        y_pred = []
        for batch in loader:
            out = model(
                batch.x,
                batch.edge_index,
                batch.batch,
                batch.ligand_idx,
                batch.additive_idx,
                batch.base_idx,
                batch.aryl_idx,
            )
            y_pred += out.view(-1).tolist()
        preds.append(y_pred)

    return torch.tensor(preds)
