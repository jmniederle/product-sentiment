import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchtext.vocab import GloVe

from sentiment_model.data_utils.metrics import accuracy, MAE
from sentiment_model.data_utils.tweet_dataset import TweetDataset, pad_batch
from sentiment_model.model import SentimentNet
from utils import get_project_root
import numpy as np
from sentiment_model.model_calibration import predict, CalibratedModel
import torch.nn as nn


def run_evaluation(
        batch_size: int = 16,
        embedding_dim: int = 50,
        hidden_dim: int = 256,
        n_layers: int = 2,
        bidirectional: bool = True,
        num_classes: int = 3,
        dropout: float = 0.5,
        freeze_embed: bool = False,
        model_file="vivid-thunder-47/vivid-thunder-47-epoch-7.pth",
        dataset="sent_ex",
        decision_bound=None,
):

    # Set device:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Import GloVe Embeddings
    cache_path = os.path.join(get_project_root(), Path("sentiment_model/.vector_cache/"))
    glove_twitter = GloVe(name="twitter.27B", dim=embedding_dim, cache=cache_path)

    # Instantiate vectors and ensure a 0 vector is inserted for unknown characters and padded characters
    pre_embeds = glove_twitter.vectors
    pre_embeds = torch.cat((torch.zeros(2, pre_embeds.shape[1]), pre_embeds))

    # Load data:
    test_dataset = TweetDataset(dataset=dataset, split="test", pretrained_vecs=glove_twitter)
    valid_dataset = TweetDataset(split="valid", dataset="sent140_multi_class", pretrained_vecs=glove_twitter)

    # Create data loaders:
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=pad_batch)

    checkpoint_path = os.path.join(get_project_root(), Path("sentiment_model/checkpoints/"))
    model_path = os.path.join(checkpoint_path, Path(model_file))

    # Load checkpoint:
    checkpoint = torch.load(model_path)

    # Initialize model
    model = SentimentNet(vocab_size=len(test_dataset.vocab), embedding_dim=embedding_dim,
                         rnn_hidden_dim=hidden_dim, rnn_n_layers=n_layers, rnn_bidirectional=bidirectional,
                         dropout_rate=dropout, num_classes=num_classes, pretrained_embeddings=pre_embeds,
                         freeze_embed=freeze_embed)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)

    if dataset == "sent_ex":
        test_out = torch.tensor([]).to(device)
        test_target = torch.tensor([]).to(device)
        model.eval()
        with torch.no_grad():
            for batch_idx, (data, target, text_lengths) in enumerate(test_loader):
                data, target, text_lengths = data.to(device), target.to(device), text_lengths.to(device)

                # Forward pass
                output = model(data, text_lengths)
                output = torch.sigmoid(output)
                test_out = torch.cat((test_out, output))
                test_target = torch.cat((test_target, target))

        acc = accuracy(test_out, test_target)
        print(f"Test set accuracy: {acc}")
        return test_out.cpu().numpy(), test_target.cpu().numpy(), model

    elif dataset == "sent140":

        print("Calibrating model")

        # Calibrate model
        CM = CalibratedModel(model)
        CM.fit(valid_dataset, valid_dataset.get_y())

        sm = nn.Softmax(dim=1)
        test_out = torch.tensor([]).to(device)

        # model.eval()
        # with torch.no_grad():
        #     for batch_idx, (data, target, text_lengths) in enumerate(test_loader):
        #         data, target, text_lengths = data.to(device), target.to(device), text_lengths.to(device)
        #
        #         # Forward pass
        #         output = model(data, text_lengths)
        #         prob_out = sm(output)
        #         test_out = torch.cat((test_out, prob_out))
        #         test_target = torch.cat((test_target, target))
        test_target = torch.tensor(test_dataset.get_y()).to(device)
        test_out = torch.tensor(predict(CM, test_dataset, decision_bound)).to(device)

        acc = accuracy(test_out, test_target)
        print(f"Test set accuracy: {acc}")

        return test_out.cpu().numpy(), test_target.cpu().numpy(), model, CM


if __name__ == "__main__":
    preds, targets, model = run_evaluation(model_file="electric-surf-72/electric-surf-72-epoch-8.pth", num_classes=2,
                                    dataset="sent140")

    # Stellar feather is a run trained on multiclass approach sent140