"""Evaluation script for measuring mean squared error."""
import json
import logging
import pathlib
import os
import argparse
import sys

import pandas as pd
import numpy as np

import tarfile

#import sagemaker_containers
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch.utils.data.distributed
from torch.utils.data import Dataset

from sklearn.metrics import (accuracy_score, 
    precision_score, recall_score, f1_score)

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))

# Dataset class to handle data loading
class RainDataset(Dataset):
    def __init__(self, csv_file, root_dir):
        self.data = pd.read_csv(os.path.join(root_dir, csv_file))
        self.labels = np.asarray(self.data.iloc[:, 0])
    
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        single_label = self.labels[idx]

        data_array = np.asarray(self.data.iloc[idx, 1:])
        feature_tensor = torch.tensor(data_array, dtype=torch.float32)

        return (feature_tensor, np.float32(single_label))


#  Simple ANN for binary classification
# Who cares this is a demo for sagemaker not a DL demo

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
        return torch.sigmoid(self.out(x))


def _get_test_data_loader(test_batch_size, test_dir, test_file="test.csv", **kwargs):
    logger.info("Get test data loader")
    dataset = RainDataset(
        test_file,
        test_dir
    )
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=test_batch_size,
        shuffle=True,
        **kwargs
    )

def format_target(target):
    return target.to("cpu").squeeze().numpy().round().astype(np.uint)

def test(model, test_loader, device):
    model.eval()
    test_loss = 0
    correct = 0
    result_dict = {"accuracy":[], "precision":[], "recall":[], "f1_score":[]}
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.binary_cross_entropy(output.squeeze(), target.squeeze(), reduction="sum").item()  # sum up batch loss
            result_dict["accuracy"].append(accuracy_score(format_target(target), format_target(output)))
            result_dict["precision"].append(precision_score(format_target(target), format_target(output)))
            result_dict["recall"].append(recall_score(format_target(target), format_target(output)))
            result_dict["f1_score"].append(f1_score(format_target(target), format_target(output)))

    for key in result_dict.keys():
        mean = sum(result_dict[key]) / len(result_dict[key])
        result_dict[key] = mean
            
    test_loss /= len(test_loader.dataset)
    logger.info(
        "Test set stats: \n Num examples: {},  \n {}".format(result_dict, result_dict)
    )
    return {"regression_metrics": result_dict}


def model_fn(model_dir):
    model = Net()
    model = torch.nn.DataParallel(model)
    with tarfile.open(os.path.join(model_dir, 'model.tar.gz'), "r:gz") as tar:
        tar.extractall(".")
    with open(os.path.join('model.pth'), 'rb') as f:
        model.load_state_dict(torch.load(f))
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=1000,
        metavar="N",
        help="input batch size for testing (default: 1000)",
    )
    args = parser.parse_args()
    num_gpus = torch.cuda.device_count()

    use_cuda = num_gpus > 0
    logger.debug("Number of gpus available - {}".format(num_gpus))
    kwargs = {"num_workers": 1, "pin_memory": True} if use_cuda else {}
    device = torch.device("cuda" if use_cuda else "cpu")

    logger.debug("Loading Model...")
    model_path = "/opt/ml/processing/model/"
    model = model_fn(model_dir=model_path).to(device)
    # if num_gpus > 1:
    # model = torch.nn.DataParallel(model)

    logger.debug("Loading DataLoader...")
    validation_folder = "/opt/ml/processing/validation"
    test_loader = _get_test_data_loader(args.test_batch_size, validation_folder, "validation.csv")

    logger.debug("Testing!!")
    result_dict = test(model, test_loader, device)

    output_dir = "/opt/ml/processing/evaluation"
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    evaluation_path = f"{output_dir}/evaluation.json"
    with open(evaluation_path, "w") as f:
        f.write(json.dumps(result_dict))