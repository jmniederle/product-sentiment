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
from sklearn.metrics import accuracy_score
from utils import pickle_save


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
        dataset="sent140_multi_class",
        decision_bound=None,
        load_calib_model=True
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
    valid_dataset = TweetDataset(split="valid", dataset=dataset, pretrained_vecs=glove_twitter)

    # Create data loaders:
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=pad_batch)

    checkpoint_path = os.path.join(get_project_root(), Path("sentiment_model/checkpoints/"))
    model_path = os.path.join(checkpoint_path, Path(model_file))

    # Load checkpoint:
    checkpoint = torch.load(model_path)

    if not load_calib_model:
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

    elif dataset == "sent140_multi_class":
        valid_y = valid_dataset.get_y()
        calib_save_folder = os.path.join(get_project_root(), Path("sentiment_model/checkpoints/calibrated_model/"))

        if load_calib_model:
            model_args = {"vocab_size": len(test_dataset.vocab), "embedding_dim": embedding_dim,
                          "rnn_hidden_dim": hidden_dim, "rnn_n_layers": n_layers, "rnn_bidirectional": bidirectional,
                          "dropout_rate": dropout, "num_classes": num_classes, "pretrained_embeddings": pre_embeds,
                          "freeze_embed": freeze_embed}

            print("Loading calibrated model")
            CM = CalibratedModel(None)
            CM.load(calib_save_folder, model_args)

        else:
            # Calibrate model
            print("Calibrating model")
            CM = CalibratedModel(model)
            CM.fit(valid_dataset, valid_y)
            print(calib_save_folder)
            CM.save(calib_save_folder)

        test_target = torch.tensor(test_dataset.get_y()).to(device)
        test_out = torch.tensor(predict(CM, test_dataset, decision_bound)).to(device)

        test_out, test_target = test_out.cpu().numpy(), test_target.cpu().numpy()

        acc = accuracy_score(test_out, test_target)
        print(f"Test set accuracy: {acc}")

        # if calib_model_path is not None:
        #     calib_save_path = os.path.join(get_project_root(), Path("sentiment_model/checkpoints/"))
        #     calib_save_path = os.path.join(calib_save_path, calib_model_path)
        #     pickle_save(CM, calib_save_path)

        return test_out, test_target, CM


if __name__ == "__main__":
    preds, targets, CM = run_evaluation(model_file="lemon-forest-81/lemon-forest-81-epoch-1.pth", num_classes=2,
                                        dataset="sent140_multi_class", decision_bound=(0.5208333333333334, 0.625),
                                        load_calib_model=True)
    # Stellar feather is a run trained on multiclass approach sent140
