import numpy as np
import torch
from sklearn.preprocessing import OneHotEncoder

def data_feeder(
        tokenizer,
        x_data: np.ndarray,
        y_data: np.ndarray,
        device: str = 'cuda',
        tensor_type: str = 'pt'
):
    def convert_to_one_hot(data: np.ndarray):
        encoder = OneHotEncoder(sparse=False)
        encoded_vals = encoder.fit_transform(data.reshape(-1, 1))
        return encoded_vals

    idx = 0
    encoded_y = convert_to_one_hot(y_data)
    while True:
        if idx >= len(x_data):
            break
        to_encode = x_data[idx]
        y_vals = torch.tensor([encoded_y[idx]]).float().to(device)
        tokenized_data = tokenizer(to_encode, return_tensors=tensor_type).to(device)
        idx += 1
        yield tokenized_data, y_vals
