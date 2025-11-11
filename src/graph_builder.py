from rdkit import Chem
from rdkit.Chem import AllChem
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

def bond_features(bond):
    """Extract bond features for edge attributes."""
    bond_type_map = {
        Chem.BondType.SINGLE: 1.0,
        Chem.BondType.DOUBLE: 2.0,
        Chem.BondType.TRIPLE: 3.0,
        Chem.BondType.AROMATIC: 1.5,
    }
    return torch.tensor([
        bond_type_map.get(bond.GetBondType(), 0.0),  # Bond order
        int(bond.GetIsConjugated()),                  # Conjugation
        int(bond.IsInRing()),                         # Ring membership
        int(bond.GetIsAromatic()),                    # Aromaticity
    ], dtype=torch.float)

def get_3d_coords(mol):
    """Generate 3D conformer and return coordinates."""
    try:
        mol_copy = Chem.Mol(mol)
        # Add explicit hydrogens to avoid warnings
        mol_copy = Chem.AddHs(mol_copy)
        
        # Generate 3D conformer with explicit Hs
        result = AllChem.EmbedMolecule(mol_copy, randomSeed=42)
        if result == -1:
            # Embedding failed, return zeros
            return torch.zeros((mol.GetNumAtoms(), 3), dtype=torch.float)
        
        # Optimize with MMFF force field
        AllChem.MMFFOptimizeMolecule(mol_copy)
        
        # Remove Hs before extracting coordinates (we only want heavy atoms)
        mol_copy = Chem.RemoveHs(mol_copy)
        conformer = mol_copy.GetConformer()
        
        coords = torch.tensor([
            [conformer.GetAtomPosition(i).x,
             conformer.GetAtomPosition(i).y,
             conformer.GetAtomPosition(i).z]
            for i in range(mol_copy.GetNumAtoms())
        ], dtype=torch.float)
        return coords
    except:
        # Fallback to zeros if 3D generation fails
        return torch.zeros((mol.GetNumAtoms(), 3), dtype=torch.float)

def mol_to_graph(smiles: str, y: torch.Tensor, use_3d=False):
    """
    Convert SMILES string to PyTorch Geometric graph.
    
    Args:
        smiles: SMILES representation of molecule(s)
        y: Target yield value
        use_3d: Whether to generate 3D coordinates (slower, set False by default)
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # Node feature matrix
    x = torch.stack([atom_features(atom) for atom in mol.GetAtoms()], dim=0)

    # Edge index and edge attributes for bonds
    edge_index = [[], []]
    edge_attr = []
    
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edge_index[0] += [i, j]
        edge_index[1] += [j, i]
        
        # Add edge features for both directions
        bond_feat = bond_features(bond)
        edge_attr.append(bond_feat)
        edge_attr.append(bond_feat)

    edge_index = torch.tensor(edge_index, dtype=torch.long)
    edge_attr = torch.stack(edge_attr, dim=0) if edge_attr else torch.empty((0, 4))
    
    # 3D conformer coordinates (optional, can be slow)
    pos = get_3d_coords(mol) if use_3d else None

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, pos=pos, y=y)
