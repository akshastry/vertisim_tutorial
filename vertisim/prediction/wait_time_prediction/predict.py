import torch
from model import PassengerWaitTimeLSTM

def load_model(model_path, input_size, hidden_layer_size, output_size):
    model = PassengerWaitTimeLSTM(input_size, hidden_layer_size, output_size)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def predict(model, input_sequence):
    with torch.no_grad():
        prediction = model(input_sequence)
        return prediction.item()

# Example usage
# model = load_model('lstm_model.pth', 5, 100, 1)
# prediction = predict(model, your_input_sequence)
