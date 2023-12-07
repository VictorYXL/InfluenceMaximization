import os
import datetime
import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score


class Trainer:
    def __init__(self, model, training_dataset, evaluation_dataset, device=torch.device("cpu"), learning_rate=1e-3, output_dir="output", epochs=100, batch_size=8):
        self.model = model
        self.training_dataset = training_dataset
        self.evaluation_dataset = evaluation_dataset
        self.device = device
        self.criterion = nn.BCEWithLogitsLoss()  # Binary cross-entropy with logits loss
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.output_dir = output_dir
        self.epochs = epochs
        self.batch_size = batch_size
        self.model_path = os.path.join(output_dir, "best_model.pth")
        self.model.to(self.device)

    def train(self):
        dataloader = DataLoader(self.training_dataset, batch_size=self.batch_size, shuffle=True)
        self.model.train()  # Set the model to training mode
        best_accuracy = 0

        for epoch in range(self.epochs):
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
            print(datetime.datetime.now())
            print(f"Epoch {epoch+1}/{self.epochs}, Loss: {avg_loss:.4f}")
            current_accuracy, _ = self.evaluate()
            if current_accuracy > best_accuracy:
                best_accuracy = current_accuracy
                self.save_model(self.model_path)

    def evaluate(self):
        dataloader = DataLoader(self.evaluation_dataset, batch_size=len(self.evaluation_dataset), shuffle=False)
        self.model.eval()  # Set the model to evaluation mode

        with torch.no_grad():  # Disable gradient calculation
            for inputs, labels in dataloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)

                # Since outputs are logits, apply sigmoid to convert to probabilities
                probabilities = torch.sigmoid(outputs)
                predictions = torch.round(probabilities)

                # Move predictions and labels to CPU for evaluation if necessary
                predictions = predictions.cpu().numpy()
                labels = labels.cpu().numpy()

                # Calculate overall accuracy
                accuracy = sum(sum(labels == predictions)) / (len(labels) * len(labels[0]))
                overall_accuracy = accuracy_score(labels, predictions)

                # Calculate accuracy for each class
                class_accuracies = []
                for class_index in range(2):  # Assuming binary classification (0 and 1)
                    class_mask = labels == class_index
                    class_acc = accuracy_score(labels[class_mask], predictions[class_mask])
                    class_accuracies.append(class_acc)
                print(f"Accuracy manual calculated: {accuracy}")
                print(f"Validation Accuracy (Overall): {overall_accuracy:.4f}")
                print(f"Validation Accuracy (Class 0 - Not Influenced): {class_accuracies[0]:.4f}")
                print(f"Validation Accuracy (Class 1 - Influenced): {class_accuracies[1]:.4f}")
                return accuracy, class_accuracies

    def save_model(self, file_path):
        """Save the model parameters to the specified file."""
        torch.save(self.model.state_dict(), file_path)
        print(f"Model saved to {file_path}")

    def load_model(self, file_path):
        """Load the model parameters from the specified file."""
        self.model.load_state_dict(torch.load(file_path, map_location=self.device))
        self.model.eval()  # Set the model to evaluation mode
        print(f"Model loaded from {file_path}")
