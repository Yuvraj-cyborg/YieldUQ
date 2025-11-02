from rdkit import Chem
from torch_geometric.data import Data
import torch

def mol_to_graph(smiles: str, y: float):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # Node features (atom number)
    atom_feats = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
    x = torch.tensor(atom_feats, dtype=torch.float).unsqueeze(1)

    # Edge list (bonds)
    edge_index = [[], []]
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edge_index[0] += [i, j]
        edge_index[1] += [j, i]

    edge_index = torch.tensor(edge_index, dtype=torch.long)
    y = torch.tensor([y], dtype=torch.float)

    return Data(x=x, edge_index=edge_index, y=y)
