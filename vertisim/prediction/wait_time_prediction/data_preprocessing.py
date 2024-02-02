from state_trajectory_dataset import create_sequences, StateTrajectoryDataset

seq_length = 20  # number of previous timestamps to use as input variables to predict the next timestamp
target_column_index = -1  # waiting time is the last element in the state
sequences, targets = create_sequences(data, seq_length, target_column_index)

# Normalize your sequences and targets as needed


import torch
from sklearn.preprocessing import MinMaxScaler

def normalize_data(df, feature_columns, target_column):
    scaler = MinMaxScaler()
    df[feature_columns] = scaler.fit_transform(df[feature_columns])
    return df, scaler

def create_sequences(input_data, seq_length, target_column):
    sequences = []
    targets = []
    for i in range(len(input_data) - seq_length):
        seq = input_data[i:i+seq_length][feature_columns].values
        target = input_data[i+seq_length][target_column]
        sequences.append(seq)
        targets.append(target)
    return torch.FloatTensor(sequences), torch.FloatTensor(targets)
