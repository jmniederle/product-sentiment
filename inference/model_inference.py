import torch
from sentiment_model.model import SentimentNet
from torch.nn import Softmax
from tqdm import tqdm
import numpy as np


def load_model_for_inference(model_file, checkpoint_path = "../sentiment_model/checkpoints/", device="cpu"):
    checkpoint = torch.load(str(checkpoint_path) + model_file)
    sentiment_net = SentimentNet(vocab_size=1193516, embedding_dim=50)
    sentiment_net.load_state_dict(checkpoint['model_state_dict'])
    sentiment_net.to(device)
    sentiment_net.eval()
    return sentiment_net


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
