import os
import json
import pickle

import torch
import numpy as np
from six import BytesIO

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(26, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, 16)
        self.fc4 = nn.Linear(16, 8)
        self.out = nn.Linear(8, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.dropout(x, p=0.25, training=self.training)
        x = F.relu(self.fc4(x))
        x = F.dropout(x, p=0.5, training=self.training)
        return F.sigmoid(self.out(x))


def model_fn(model_dir):
    model = Net()
    model.load_state_dict(torch.load(os.path.join(model_dir, 'model.pt')))
    model.to(device)
    model.eval()
    return model


def input_fn(request_body, request_content_type):
    """An input_fn that loads a pickled tensor"""
    if request_content_type == 'application/python-pickle':
        return torch.load(BytesIO(request_body))
    elif request_content_type == 'application/json':
        return torch.tensor(json.loads(request_body)["data"], dtype=torch.float32, device=device)
    elif request_content_type == "application/x-npy":
        return torch.from_numpy(np.load(BytesIO(request_body)))
    else:
        raise TypeError(f"Type not supported - {request_content_type}")


def predict_fn(input_data, model):
    with torch.no_grad(True):
        output = model.forward(input_data)
    return output


def output_fn(predictions, content_type):
    res = predictions.cpu().numpy().tolist()
    if content_type == "application/json":
        return json.dumps({"output": res})
    elif content_type == "application/x-npy":
        return res.cpu().numpy()
    elif content_type == 'application/python-pickle':
        return pickle.dumps(predictions.cpu().numpy().tolist())

