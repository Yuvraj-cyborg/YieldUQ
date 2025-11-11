import torch
import pandas as pd
from torch_geometric.data import InMemoryDataset
from src.graph_builder import mol_to_graph
from tqdm import tqdm
from pathlib import Path
from torch_geometric.data import Data

class ReactionDataset(InMemoryDataset):
    def __init__(self, csv_path="data/Dreher_and_Doyle_input_data.xlsx", transform=None, pre_transform=None):
        self.csv_path = Path(csv_path)
        super().__init__('.', transform, pre_transform)
        self.data, self.slices, self.y_mean, self.y_std, self.category_maps = self.process_data()

    def process_data(self):
        if not self.csv_path.exists():
            raise FileNotFoundError(f"Dataset not found at {self.csv_path.resolve()}")

        df = pd.read_excel(self.csv_path)
        print(f"Loaded dataset with columns: {list(df.columns)}")
        print(f"Total rows: {len(df)}")

        expected = ['Ligand', 'Additive', 'Base', 'Aryl halide', 'Output']
        missing = [c for c in expected if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns in dataset: {missing}")

        df = df.dropna(subset=expected)

        # Combine reaction components into a single representation string
        df['combined'] = (
            df['Ligand'].astype(str) + '.' +
            df['Additive'].astype(str) + '.' +
            df['Base'].astype(str) + '.' +
            df['Aryl halide'].astype(str)
        )

        # Normalize target yield
        y_min, y_max = df["Output"].min(), df["Output"].max()
        df["yield_norm"] = (df["Output"] - y_min) / (y_max - y_min)

        # Map categorical columns to integer indices
        category_maps = {}
        for col in ['Ligand', 'Additive', 'Base', 'Aryl halide']:
            uniques = sorted(df[col].unique())
            category_maps[col] = {v: i for i, v in enumerate(uniques)}
            df[f"{col}_idx"] = df[col].map(category_maps[col])

        data_list = []
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing reactions"):
            y = torch.tensor([row['yield_norm']], dtype=torch.float)
            
            # Try to parse the combined SMILES string into a graph
            # Note: use_3d=False by default (3D generation is slow and causes warnings)
            # Set use_3d=True if you need 3D coordinates for future enhancements
            combined_smiles = row['combined']
            graph_data = mol_to_graph(combined_smiles, y, use_3d=False)
            
            if graph_data is None:
                # Fallback: create minimal dummy graph if parsing fails
                x = torch.ones((1, 6))
                edge_index = torch.zeros((2, 0), dtype=torch.long)
                edge_attr = torch.zeros((0, 4), dtype=torch.float)
                pos = None  # No 3D coords in fallback
                
                graph_data = Data(
                    x=x,
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                    pos=pos,
                    y=y
                )
            
            # Add categorical indices as attributes
            graph_data.ligand_idx = torch.tensor(row["Ligand_idx"])
            graph_data.additive_idx = torch.tensor(row["Additive_idx"])
            graph_data.base_idx = torch.tensor(row["Base_idx"])
            graph_data.aryl_idx = torch.tensor(row["Aryl halide_idx"])
            
            data_list.append(graph_data)

        data, slices = self.collate(data_list)
        return data, slices, y_min, y_max, category_maps



if __name__ == "__main__":
    print("Loading reaction dataset...")
    dataset = ReactionDataset("data/Dreher_and_Doyle_input_data.xlsx")
    print(f"Dataset loaded successfully.")
    print(f"Total samples: {len(dataset)}")
    print(f"Example graph data object:\n{dataset[0]}")
