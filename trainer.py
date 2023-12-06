import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader


class Trainer:
    def __init__(self, model, training_dataset, evaluation_dataset, device=torch.device("cpu"), learning_rate=1e-3):
        self.model = model
        self.training_dataset = training_dataset
        self.evaluation_dataset = evaluation_dataset
        self.device = device
        self.criterion = nn.BCEWithLogitsLoss()  # Binary cross-entropy with logits loss
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    def train(self, epochs, batch_size):
        dataloader = DataLoader(self.training_dataset, batch_size=batch_size, shuffle=True)
        self.model.train()  # Set the model to training mode

        for epoch in range(epochs):
            total_loss = 0.0
            for inputs, labels in dataloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()  # Zero the parameter gradients
                outputs = self.model(inputs)  # Forward pass
                loss = self.criterion(outputs, labels)  # Compute the loss
                loss.backward()  # Backward pass
                self.optimizer.step()  # Optimize the model

                total_loss += loss.item()

            avg_loss = total_loss / len(dataloader)
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
            self.evaluate()
  
    def evaluate(self):  
        dataloader = DataLoader(self.evaluation_dataset, batch_size=len(self.evaluation_dataset), shuffle=False)  
        self.model.eval()  # Set the model to evaluation mode  
          
        with torch.no_grad():  # Disable gradient calculation  
            for inputs, labels in dataloader:  
                inputs, labels = inputs.to(self.device), labels.to(self.device)  
                outputs = self.model(inputs)  
                  
                # Apply sigmoid and round to get binary class predictions  
                # Since outputs are logits, apply sigmoid to convert to probabilities  
                probabilities = torch.sigmoid(outputs)  
                predictions = torch.round(probabilities)  
                  
                # Move predictions and labels to CPU for evaluation if necessary  
                predictions = predictions.cpu().numpy()  
                labels = labels.cpu().numpy()  
                  
                # Calculate accuracy  
                accuracy = sum(sum(labels == predictions)) / (len(labels) * len(labels[0]))
                  
                print(f"Validation Accuracy: {accuracy:.4f}")  
                return accuracy  

    def save_model(self, file_path):
        """Save the model parameters to the specified file."""
        torch.save(self.model.state_dict(), file_path)
        print(f"Model saved to {file_path}")

    def load_model(self, file_path):
        """Load the model parameters from the specified file."""
        self.model.load_state_dict(torch.load(file_path, map_location=self.device))
        self.model.eval()  # Set the model to evaluation mode
        print(f"Model loaded from {file_path}")
