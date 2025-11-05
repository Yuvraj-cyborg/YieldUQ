import torch
from torch_geometric.nn import GCNConv, global_mean_pool


class GNNModel(torch.nn.Module):
    def __init__(self, in_channels=6, hidden_channels=64, category_sizes=None, embed_dim=16):
        super(GNNModel, self).__init__()
        self.category_sizes = category_sizes or {}
        self.embed_dim = embed_dim

        # Embeddings for categorical features
        self.emb_ligand = torch.nn.Embedding(self.category_sizes.get("Ligand", 1), embed_dim)
        self.emb_additive = torch.nn.Embedding(self.category_sizes.get("Additive", 1), embed_dim)
        self.emb_base = torch.nn.Embedding(self.category_sizes.get("Base", 1), embed_dim)
        self.emb_aryl = torch.nn.Embedding(self.category_sizes.get("Aryl halide", 1), embed_dim)

        # GNN layers
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)

        # Fully connected head
        total_in = hidden_channels + 4 * embed_dim
        self.lin1 = torch.nn.Linear(total_in, hidden_channels)
        self.lin2 = torch.nn.Linear(hidden_channels, 1)

        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(0.2)

    def forward(self, x, edge_index, batch, ligand_idx, additive_idx, base_idx, aryl_idx):
        # Graph feature extraction
        x = self.relu(self.conv1(x, edge_index))
        x = self.relu(self.conv2(x, edge_index))

        # Pool node features per reaction
        x = global_mean_pool(x, batch)

        # Categorical embeddings
        ligand_emb = self.emb_ligand(ligand_idx)
        add_emb = self.emb_additive(additive_idx)
        base_emb = self.emb_base(base_idx)
        aryl_emb = self.emb_aryl(aryl_idx)

        # Combine all
        cat_embed = torch.cat([ligand_emb, add_emb, base_emb, aryl_emb], dim=-1)
        x = torch.cat([x, cat_embed], dim=-1)

        # MLP head
        x = self.dropout(self.relu(self.lin1(x)))
        out = self.lin2(x)
        return out
