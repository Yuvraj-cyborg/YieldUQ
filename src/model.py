import torch
from torch_geometric.nn import GCNConv, GINEConv, global_mean_pool, GlobalAttention, Set2Set
import torch.nn.functional as F


class GNNModel(torch.nn.Module):
    """Enhanced GNN with edge features, attention pooling, and heteroscedastic uncertainty."""
    
    def __init__(self, in_channels=6, edge_dim=4, hidden_channels=64, 
                 category_sizes=None, embed_dim=16, pooling='attention',
                 use_edge_features=True, heteroscedastic=True, num_layers=3):
        super(GNNModel, self).__init__()
        self.category_sizes = category_sizes or {}
        self.embed_dim = embed_dim
        self.pooling_type = pooling
        self.use_edge_features = use_edge_features
        self.heteroscedastic = heteroscedastic
        self.num_layers = num_layers

        # Embeddings for categorical features
        self.emb_ligand = torch.nn.Embedding(self.category_sizes.get("Ligand", 1), embed_dim)
        self.emb_additive = torch.nn.Embedding(self.category_sizes.get("Additive", 1), embed_dim)
        self.emb_base = torch.nn.Embedding(self.category_sizes.get("Base", 1), embed_dim)
        self.emb_aryl = torch.nn.Embedding(self.category_sizes.get("Aryl halide", 1), embed_dim)

        # GNN layers with edge features
        self.convs = torch.nn.ModuleList()
        if use_edge_features:
            # GINEConv with edge features
            nn1 = torch.nn.Sequential(
                torch.nn.Linear(in_channels, hidden_channels),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_channels, hidden_channels)
            )
            self.convs.append(GINEConv(nn1, edge_dim=edge_dim, train_eps=True))
            
            for _ in range(num_layers - 1):
                nn_layer = torch.nn.Sequential(
                    torch.nn.Linear(hidden_channels, hidden_channels),
                    torch.nn.ReLU(),
                    torch.nn.Linear(hidden_channels, hidden_channels)
                )
                self.convs.append(GINEConv(nn_layer, edge_dim=edge_dim, train_eps=True))
        else:
            # Standard GCNConv without edge features
            self.convs.append(GCNConv(in_channels, hidden_channels))
            for _ in range(num_layers - 1):
                self.convs.append(GCNConv(hidden_channels, hidden_channels))

        # Advanced pooling
        if pooling == 'attention':
            gate_nn = torch.nn.Sequential(
                torch.nn.Linear(hidden_channels, hidden_channels),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_channels, 1)
            )
            self.pool = GlobalAttention(gate_nn)
            pool_out_dim = hidden_channels
        elif pooling == 'set2set':
            self.pool = Set2Set(hidden_channels, processing_steps=3)
            pool_out_dim = 2 * hidden_channels
        else:  # mean pooling
            self.pool = None
            pool_out_dim = hidden_channels

        # Fully connected head
        total_in = pool_out_dim + 4 * embed_dim
        self.lin1 = torch.nn.Linear(total_in, hidden_channels)
        
        # Heteroscedastic output: predict both mean and log variance
        if heteroscedastic:
            self.lin_mean = torch.nn.Linear(hidden_channels, 1)
            self.lin_logvar = torch.nn.Linear(hidden_channels, 1)
        else:
            self.lin2 = torch.nn.Linear(hidden_channels, 1)

        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(0.2)

    def forward(self, x, edge_index, batch, ligand_idx, additive_idx, base_idx, aryl_idx, 
                edge_attr=None, return_uncertainty=False):
        # Graph feature extraction with edge features
        for i, conv in enumerate(self.convs):
            if self.use_edge_features and edge_attr is not None:
                x = conv(x, edge_index, edge_attr)
            else:
                x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = self.relu(x)

        # Pool node features per reaction
        if self.pool is not None:
            x = self.pool(x, batch)
        else:
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
        
        # Heteroscedastic output
        if self.heteroscedastic:
            mean = self.lin_mean(x)
            log_var = self.lin_logvar(x)
            if return_uncertainty:
                return mean, log_var
            return mean  # Default: just return mean for compatibility
        else:
            out = self.lin2(x)
            return out


class EnsembleModel(torch.nn.Module):
    """Ensemble of multiple GNN models for better uncertainty estimation."""
    
    def __init__(self, num_models=5, **model_kwargs):
        super(EnsembleModel, self).__init__()
        self.models = torch.nn.ModuleList([
            GNNModel(**model_kwargs) for _ in range(num_models)
        ])
        self.num_models = num_models
    
    def forward(self, *args, return_all=False, **kwargs):
        outputs = [model(*args, **kwargs) for model in self.models]
        
        if return_all:
            return torch.stack(outputs, dim=0)
        else:
            # Return mean prediction
            return torch.stack(outputs, dim=0).mean(dim=0)
    
    def predict_with_uncertainty(self, *args, **kwargs):
        """Get mean and std across ensemble."""
        outputs = torch.stack([model(*args, **kwargs) for model in self.models], dim=0)
        mean = outputs.mean(dim=0)
        std = outputs.std(dim=0)
        return mean, std
