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

# Model setup
model = GNNModel(in_channels=1, hidden_channels=64)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = torch.nn.MSELoss()

# Training
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
