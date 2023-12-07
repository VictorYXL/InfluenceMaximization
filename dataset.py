import os
import torch
from torch.utils.data import Dataset
import json
from graph_loader import load_graph_from_tsv
from scenarios.independent_cascade.ic_simulator import ic_simulator
from scenarios.linear_threshold.lt_simulator import lt_simulator

class InfluenceDataset(Dataset):
    def __init__(self, dataset_path: str, mode: str):
        """Load the dataset from the JSON file"""
        with open(dataset_path, 'r') as file:
            json_data = json.load(file)
        self.graph = load_graph_from_tsv(json_data["node_file"], json_data["edge_file"])
        self.influence_function = eval(json_data["influence_function"])
        self._data = json_data[mode]
   
    def __len__(self):
        # Return the number of samples in the dataset
        return len(self._data)

    def __getitem__(self, idx):
        # Retrieve the sample at the specified index
        sample = self._data[idx]
       
        # Convert the input and label lists to PyTorch tensors
        input_tensor = torch.tensor(sample['input'], dtype=torch.float32)
        label_tensor = torch.tensor(sample['label'], dtype=torch.float32)
        
        # Return the sample as a tuple of input and label tensors
        return input_tensor, label_tensor

if __name__ == "__main__":
    # Usage example:
    dataset_path = os.path.join("graph", "example", "dataset.json")
    dataset = InfluenceDataset(dataset_path)

    # Access the first sample in the dataset
    print(dataset.graph)
    print(dataset.influence_function)
    print(dataset[0])
