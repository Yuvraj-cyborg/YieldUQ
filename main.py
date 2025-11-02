import torch
from torch_geometric.loader import DataLoader
from src.dataset import ReactionDataset
from src.model import GNNModel
from src.train import train_epoch, evaluate, mc_dropout_predict
from src.evaluate import analyze_uncertainty
from src.utils import set_seed
from sklearn.model_selection import train_test_split

def main():
    set_seed()
    dataset = ReactionDataset("data/bh_yield.csv")
    train_idx, test_idx = train_test_split(range(len(dataset)), test_size=0.2, random_state=42)
    train_data = [dataset[i] for i in train_idx]
    test_data = [dataset[i] for i in test_idx]

    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=32)

    model = GNNModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = torch.nn.MSELoss()

    for epoch in range(20):
        loss = train_epoch(model, train_loader, optimizer, loss_fn)
        r2 = evaluate(model, test_loader)
        print(f"Epoch {epoch+1:02d} | Loss: {loss:.4f} | R²: {r2:.3f}")

    # MC dropout inference
    mc_preds = mc_dropout_predict(model, test_loader, n_samples=10)
    y_true = [b.y.item() for b in test_data]
    analyze_uncertainty(mc_preds, y_true)

if __name__ == "__main__":
    main()
