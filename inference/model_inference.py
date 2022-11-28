import torch
from sentiment_model.model import SentimentNet
from torch.nn import Softmax
from tqdm import tqdm
import numpy as np
from utils import get_project_root
from pathlib import Path
import os
from sentiment_model.model_calibration import CalibratedModel


def load_model_for_inference(model_file=None, model_args=None, device="cpu", calibrated=False):
    if not calibrated:
        checkpoint_path = os.path.join(get_project_root(), Path("sentiment_model/checkpoints/"))
        model_path = os.path.join(checkpoint_path, Path(model_file))
        checkpoint = torch.load(model_path)
        sentiment_net = SentimentNet(**model_args)
        sentiment_net.load_state_dict(checkpoint['model_state_dict'])
        sentiment_net.to(device)
        sentiment_net.eval()
        return sentiment_net

    else:
        calib_save_folder = os.path.join(get_project_root(), Path("sentiment_model/checkpoints/calibrated_model/"))
        if model_file is not None:
            calib_save_folder = os.path.join(calib_save_folder, Path(model_file))

        CM = CalibratedModel(None)
        CM.load(calib_save_folder, model_args)
        return CM


def run_inference(model, data_loader, device='cpu'):

    results = []

    soft_max = Softmax(dim=1)

    with torch.no_grad():
        p_bar = tqdm(total=len(data_loader.dataset))
        for batch_idx, (data, text_lengths) in enumerate(data_loader):
            # For assignment 3.2, we need to know the lengths of the targets

            data, text_lengths = data.to(device), text_lengths.to(device)

            # Forward pass
            output = model(data, text_lengths)
            prob_out = soft_max(output, )
            results.extend(prob_out.tolist())
            p_bar.update(data.shape[0])

    return np.array(results)
