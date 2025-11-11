import torch
from tqdm import tqdm
import os


def heteroscedastic_loss(mean, log_var, target):
    """Negative log-likelihood loss for heteroscedastic uncertainty."""
    precision = torch.exp(-log_var)
    loss = 0.5 * precision * (target - mean) ** 2 + 0.5 * log_var
    return loss.mean()


def train_epoch(model, loader, optimizer, loss_fn='mse', use_heteroscedastic=False):
    model.train()
    total_loss = 0
    for batch in tqdm(loader, desc="Training", leave=False):
        optimizer.zero_grad()

        # Handle edge attributes
        edge_attr = batch.edge_attr if hasattr(batch, 'edge_attr') else None
        
        if use_heteroscedastic:
            mean, log_var = model(
                batch.x,
                batch.edge_index,
                batch.batch,
                batch.ligand_idx,
                batch.additive_idx,
                batch.base_idx,
                batch.aryl_idx,
                edge_attr=edge_attr,
                return_uncertainty=True
            )
            loss = heteroscedastic_loss(mean.view(-1), log_var.view(-1), batch.y.view(-1))
        else:
            out = model(
                batch.x,
                batch.edge_index,
                batch.batch,
                batch.ligand_idx,
                batch.additive_idx,
                batch.base_idx,
                batch.aryl_idx,
                edge_attr=edge_attr,
            )
            loss = loss_fn(out.view(-1), batch.y.view(-1))
        
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, use_heteroscedastic=False):
    model.eval()
    y_true, y_pred = [], []
    for batch in tqdm(loader, desc="Evaluating", leave=False):
        edge_attr = batch.edge_attr if hasattr(batch, 'edge_attr') else None
        
        if use_heteroscedastic:
            mean, _ = model(
                batch.x,
                batch.edge_index,
                batch.batch,
                batch.ligand_idx,
                batch.additive_idx,
                batch.base_idx,
                batch.aryl_idx,
                edge_attr=edge_attr,
                return_uncertainty=True
            )
            out = mean
        else:
            out = model(
                batch.x,
                batch.edge_index,
                batch.batch,
                batch.ligand_idx,
                batch.additive_idx,
                batch.base_idx,
                batch.aryl_idx,
                edge_attr=edge_attr,
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
def mc_dropout_predict(model, loader, n_samples=50):
    """Enhanced MC Dropout with more samples."""
    model.train()  # enable dropout
    preds = []

    for _ in tqdm(range(n_samples), desc="MC Sampling", leave=False):
        y_pred = []
        for batch in loader:
            edge_attr = batch.edge_attr if hasattr(batch, 'edge_attr') else None
            out = model(
                batch.x,
                batch.edge_index,
                batch.batch,
                batch.ligand_idx,
                batch.additive_idx,
                batch.base_idx,
                batch.aryl_idx,
                edge_attr=edge_attr,
            )
            y_pred += out.view(-1).tolist()
        preds.append(y_pred)

    return torch.tensor(preds)


@torch.no_grad()
def heteroscedastic_predict(model, loader):
    """Predict with heteroscedastic uncertainty (single forward pass)."""
    model.eval()
    means, log_vars = [], []
    
    for batch in tqdm(loader, desc="Heteroscedastic Prediction", leave=False):
        edge_attr = batch.edge_attr if hasattr(batch, 'edge_attr') else None
        mean, log_var = model(
            batch.x,
            batch.edge_index,
            batch.batch,
            batch.ligand_idx,
            batch.additive_idx,
            batch.base_idx,
            batch.aryl_idx,
            edge_attr=edge_attr,
            return_uncertainty=True
        )
        means += mean.view(-1).tolist()
        log_vars += log_var.view(-1).tolist()
    
    means = torch.tensor(means)
    stds = torch.exp(0.5 * torch.tensor(log_vars))  # Convert log_var to std
    return means, stds


@torch.no_grad()
def ensemble_predict(ensemble_model, loader):
    """Predict with ensemble uncertainty."""
    ensemble_model.eval()
    all_preds = []
    
    for batch in tqdm(loader, desc="Ensemble Prediction", leave=False):
        edge_attr = batch.edge_attr if hasattr(batch, 'edge_attr') else None
        outputs = ensemble_model(
            batch.x,
            batch.edge_index,
            batch.batch,
            batch.ligand_idx,
            batch.additive_idx,
            batch.base_idx,
            batch.aryl_idx,
            edge_attr=edge_attr,
            return_all=True
        )
        all_preds.append(outputs)
    
    # Concatenate all batches
    all_preds = torch.cat([p.transpose(0, 1) for p in all_preds], dim=0)  # [num_samples, num_models]
    means = all_preds.mean(dim=1)
    stds = all_preds.std(dim=1)
    return means, stds


def save_checkpoint(model, optimizer, epoch, r2, path='checkpoints'):
    """Save model checkpoint."""
    os.makedirs(path, exist_ok=True)
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'r2': r2,
    }
    filepath = os.path.join(path, f'model_epoch_{epoch}_r2_{r2:.4f}.pt')
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved: {filepath}")


def load_checkpoint(model, optimizer, filepath):
    """Load model checkpoint."""
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    r2 = checkpoint['r2']
    print(f"Loaded checkpoint from epoch {epoch} with R²={r2:.4f}")
    return model, optimizer, epoch, r2
