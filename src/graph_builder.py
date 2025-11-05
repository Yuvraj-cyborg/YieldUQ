from rdkit import Chem
from torch_geometric.data import Data
import torch

def atom_features(atom):
    """Create a richer feature vector for each atom."""
    return torch.tensor([
        atom.GetAtomicNum(),           # 0: atomic number
        atom.GetDegree(),              # 1: number of bonded neighbors
        atom.GetTotalValence(),        # 2: total valence
        atom.GetFormalCharge(),        # 3: formal charge
        atom.GetTotalNumHs(),          # 4: hydrogen count
        int(atom.GetIsAromatic()),     # 5: aromaticity flag
    ], dtype=torch.float)

def mol_to_graph(smiles: str, y: torch.Tensor):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # Node feature matrix
    x = torch.stack([atom_features(atom) for atom in mol.GetAtoms()], dim=0)

    # Edge index for bonds
    edge_index = [[], []]
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edge_index[0] += [i, j]
        edge_index[1] += [j, i]

    edge_index = torch.tensor(edge_index, dtype=torch.long)

    return Data(x=x, edge_index=edge_index, y=y)
