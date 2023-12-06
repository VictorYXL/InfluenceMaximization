import os
import torch
from dataset import InfluenceDataset
from scenarios.linear_threshold.lt_model import LTNet
from trainer import Trainer


data_file = os.path.join("data", "example", "dataset", "dataset_100_30.json")
training_dataset = InfluenceDataset(data_file, "training_data")
evaluation_dataset = InfluenceDataset(data_file, "evaluation_data")
model = LTNet(len(training_dataset.graph.nodes))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  
trainer = Trainer(model, training_dataset, evaluation_dataset, learning_rate=1e-3)

trainer.train(epochs=1000, batch_size=2)

trainer.save_model(os.path.join("data", "example", "model", "model_100.pth"))
print(list(model.parameters())[0])
print(list(model.parameters())[1])
