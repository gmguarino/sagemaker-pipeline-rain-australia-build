import torch
import numpy as np

def model_fn():
    pass

def input_fn(request_body, request_content_type):
    pass

def predict_fn(input_data, model):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    with torch.no_grad():
        return model(input_data.to(device))

def output_fn(prediction, content_type):
    pass