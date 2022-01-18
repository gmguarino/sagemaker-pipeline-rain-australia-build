import os
import json

import torch
import numpy as np
from six import BytesIO


def model_fn(model_dir):
    model = torch.jit.load(os.path.join(model_dir, 'model.pt'), map_location=torch.device('cpu'))
    if int(torch.__version__.split["."][1]) >= 5 and int(torch.__version__.split["."][2]) >= 1:
        import torcheia
        model = model.eval()
        model = torcheia.jit.attach_eia(model, 0)
    return model


def input_fn(request_body, request_content_type):
    """An input_fn that loads a pickled tensor"""
    if request_content_type == 'application/python-pickle':
        return torch.load(BytesIO(request_body))
    elif request_content_type == 'application/json':
        return torch.Tensor(json.loads(request_body)["data"])
    elif request_content_type == "application/x-npy":
        return torch.from_numpy(np.load(BytesIO(request_body)))
    else:
        raise TypeError(f"Type not supported - {request_content_type}")


def predict_fn(input_data, model):
    device = torch.device("cpu")
    input_data = input_data.to(device)
    # make sure torcheia is imported so that Elastic Inference api call will be invoked
    import torcheia
    # we need to set the profiling executor for EIA
    torch._C._jit_set_profiling_executor(False)
    with torch.jit.optimized_execution(True):
        output = model.forward(input_data)
    return output
