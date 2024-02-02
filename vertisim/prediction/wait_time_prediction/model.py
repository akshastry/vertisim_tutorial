import torch
import torch.nn as nn

class PassengerWaitTimeLSTM(nn.Module):
    def __init__(self, input_size, hidden_layer_size, output_size, num_layers):
        super(PassengerWaitTimeLSTM, self).__init__()
        self.hidden_layer_size = hidden_layer_size
        self.num_layers = num_layers

        # Define an LSTM layer with num_layers
        self.lstm = nn.LSTM(input_size, hidden_layer_size, num_layers)
        # Define a fully connected layer for the output
        self.linear = nn.Linear(hidden_layer_size, output_size)

    def forward(self, input_seq):
        hidden_states = torch.zeros(self.num_layers, input_seq.size(0), self.hidden_layer_size)
        cell_states = torch.zeros(self.num_layers, input_seq.size(0), self.hidden_layer_size)
        lstm_out, _ = self.lstm(input_seq, (hidden_states, cell_states))
        return self.linear(lstm_out[:, -1, :])