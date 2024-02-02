# train.py
import torch
import torch.nn as nn
import torch.optim as optim
from model import PassengerWaitTimeLSTM
from data_preprocessing import load_data, preprocess_data

def train_model(train_data, input_size, hidden_layer_size, output_size, num_layers, learning_rate, epochs):
    # Initialize the model with the specified parameters
    model = PassengerWaitTimeLSTM(input_size, hidden_layer_size, output_size, num_layers)

    loss_function = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(epochs):
        for sequence, labels in train_data:
            optimizer.zero_grad()

            # Forward pass through the model
            y_pred = model(sequence)

            # Compute loss and backpropagate
            single_loss = loss_function(y_pred, labels)
            single_loss.backward()
            optimizer.step()

        if epoch % 25 == 1:
            print(f'epoch: {epoch:3} loss: {single_loss.item():10.8f}')

    print(f'epoch: {epoch:3} loss: {single_loss.item():10.10f}')

# Example usage
# Load and preprocess data
# data = load_data("your_data.csv")
# X, y = preprocess_data(data, features=["your", "feature", "columns"], target="your_target_column")

# Train the model
# train_model(zip(X, y), input_size=5, hidden_layer_size=100, output_size=1, num_layers=2, learning_rate=0.001, epochs=150)
