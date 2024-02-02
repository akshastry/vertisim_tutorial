import torch
from torch.utils.data import Dataset

class StateTrajectoryDataset(Dataset):
    def __init__(self, sequences, targets):
        self.sequences = sequences
        self.targets = targets

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, index):
        seq = self.sequences[index]
        target = self.targets[index]
        return seq, target

def create_sequences(data, seq_length, target_column_index):
    sequences = []
    targets = []
    for i in range(len(data) - seq_length):
        seq = data[i:(i+seq_length)]
        target = data[i+seq_length, target_column_index]
        sequences.append(seq)
        targets.append(target)
    return torch.FloatTensor(sequences), torch.FloatTensor(targets)
