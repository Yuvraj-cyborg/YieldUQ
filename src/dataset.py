import pandas as pd
from torch_geometric.data import InMemoryDataset
from src.graph_builder import mol_to_graph
from tqdm import tqdm
from pathlib import Path


class ReactionDataset(InMemoryDataset):
    def __init__(self, csv_path="data/Dreher_and_Doyle_input_data.xlsx", transform=None, pre_transform=None):
        self.csv_path = Path(csv_path)
        super().__init__('.', transform, pre_transform)
        self.data, self.slices = self.process_data()

    def process_data(self):
        if not self.csv_path.exists():
            raise FileNotFoundError(f"Dataset not found at {self.csv_path.resolve()}")

        # Load Excel file
        df = pd.read_excel(self.csv_path)
        print(f"✅ Loaded dataset with columns: {list(df.columns)}")
        print(f"📦 Total rows: {len(df)}")

        # Validate expected columns
        expected = ['Ligand', 'Additive', 'Base', 'Aryl halide', 'Output']
        missing = [c for c in expected if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns in dataset: {missing}")

        # Drop missing rows
        df = df.dropna(subset=expected)

        # Combine reaction components into a pseudo-SMILES-like token string
        df['combined'] = (
            df['Ligand'].astype(str) + '.' +
            df['Additive'].astype(str) + '.' +
            df['Base'].astype(str) + '.' +
            df['Aryl halide'].astype(str)
        )

        # Normalize yield (Output)
        df['yield'] = df['Output'].astype(float)

        data_list = []
        for _, row in tqdm(df.iterrows(), total=len(df), desc="🧪 Processing reactions"):
            smiles = row['combined']
            y = row['yield']
            g = mol_to_graph(smiles, y)
            if g:
                data_list.append(g)

        if not data_list:
            raise ValueError("No valid molecule graphs generated — check mol_to_graph().")

        print(f"✅ Successfully processed {len(data_list)} reactions.")
        return self.collate(data_list)


if __name__ == "__main__":
    print("🔬 Loading reaction dataset...")
    dataset = ReactionDataset("data/Dreher_and_Doyle_input_data.xlsx")
    print(f"✅ Dataset loaded successfully.")
    print(f"Total samples: {len(dataset)}")
    print(f"Example graph data object:\n{dataset[0]}")
