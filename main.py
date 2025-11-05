import torch
from torch_geometric.loader import DataLoader
from src.dataset import ReactionDataset
from src.model import GNNModel
from src.train import train_epoch, evaluate, mc_dropout_predict
from src.evaluate import analyze_uncertainty
from src.utils import set_seed
from sklearn.model_selection import train_test_split

set_seed(42)

print("Loading dataset...")
dataset = ReactionDataset("data/Dreher_and_Doyle_input_data.xlsx")

# Split dataset
train_idx, test_idx = train_test_split(range(len(dataset)), test_size=0.2, random_state=42)
train_dataset = dataset[train_idx]
test_dataset = dataset[test_idx]

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# ✅ Convert category_maps (dict of dicts) → dict of lengths for embeddings
category_sizes = {k: len(v) for k, v in dataset.category_maps.items()}
print("Category sizes:", category_sizes)

# Model setup
model = GNNModel(in_channels=6, hidden_channels=64, category_sizes=category_sizes)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = torch.nn.MSELoss()

# Training loop
n_epochs = 20
for epoch in range(n_epochs):
    loss = train_epoch(model, train_loader, optimizer, loss_fn)
    r2 = evaluate(model, test_loader)
    print(f"Epoch {epoch+1}/{n_epochs} | Loss: {loss:.4f} | R²: {r2:.4f}")

# Uncertainty Estimation
print("\nRunning Monte Carlo Dropout for Uncertainty Estimation...")
mc_preds = mc_dropout_predict(model, test_loader, n_samples=20)

# Collect true labels
y_true = []
for batch in test_loader:
    y_true += batch.y.flatten().tolist()

analyze_uncertainty(mc_preds, y_true)


from src.visualize import visualize_uncertainty, visualize_error_distribution

# Visualize results
visualize_uncertainty(y_true, mc_preds, save_path="plots/yield_uncertainty.png")
visualize_error_distribution(y_true, mc_preds)