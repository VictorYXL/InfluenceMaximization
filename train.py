import os
import torch
from dataset import InfluenceDataset
from scenarios.linear_threshold.lt_model import LTNet
from trainer import Trainer

exp_name = "exp_2"
exp_dir = os.path.join("output", exp_name)
os.makedirs(exp_dir, exist_ok=True)

data_file = os.path.join("data", "facebook", "dataset", "dataset_10k_3k_0.1.json")
training_dataset = InfluenceDataset(data_file, "training_data")
evaluation_dataset = InfluenceDataset(data_file, "evaluation_data")
model = LTNet(len(training_dataset.graph.nodes))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
trainer = Trainer(model, training_dataset, evaluation_dataset, device=device, learning_rate=1e-2, output_dir=exp_dir, epochs=1000, batch_size=32)

print(f"Device: {device}")
print(f"Dataset: {data_file}")

trainer.train()
