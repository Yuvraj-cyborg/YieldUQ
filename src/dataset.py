import pandas as pd
from torch_geometric.data import InMemoryDataset
from src.graph_builder import mol_to_graph
from tqdm import tqdm

class ReactionDataset(InMemoryDataset):
    def __init__(self, csv_path, transform=None, pre_transform=None):
        self.csv_path = csv_path
        super().__init__('.', transform, pre_transform)
        self.data, self.slices = self.process_data()

    def process_data(self):
        df = pd.read_csv(self.csv_path)
        data_list = []
        for _, row in tqdm(df.iterrows(), total=len(df)):
            smiles = row['smiles']
            y = row['yield']
            g = mol_to_graph(smiles, y)
            if g: data_list.append(g)
        return self.collate(data_list)
