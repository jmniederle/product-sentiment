from pathlib import Path

import wandb
from pathlib import Path
import os
from utils import get_project_root
import torch
import torch.nn as nn
import wandb
from torch.optim import Adam, SGD
from torch.utils.data import DataLoader
from torchtext.vocab import GloVe

from sentiment_model.data_utils.tweet_dataset import TweetDataset, pad_batch
from sentiment_model.model import SentimentNet
from sentiment_model.train import train


def run_training(
        batch_size: int = 256,
        lr: float = 0.001,
        epochs: int = 15,
        embedding_dim: int = 50,
        hidden_dim: int = 256,
        n_layers: int = 2,
        bidirectional: bool = True,
        activation_fn: str = "relu",
        dropout: float = 0.5,
        optimizer: str = "Adam",
        freeze_embed: bool = True,
        logging_freq: int = 500,
        checkpoint_path: Path = Path("checkpoints/"),
        save_checkpoint: bool = False,
        dataset_name: str = "sent140_multi_class",
        wandb_logging: bool = False,
        small_subset: int = None
):
    config = {"lr": lr,
              "epochs": epochs,
              "batch_size": batch_size,
              "optimizer": optimizer,
              "activation_fn": activation_fn,
              "embedding_dim": embedding_dim,
              "rnn_hidden_dim": hidden_dim,
              "rnn_n_layers": n_layers,
              "rnn_bidirectional": bidirectional,
              "dropout": dropout,
              "freeze_embed": freeze_embed}

    if wandb_logging:
        try:
            with open('wandb_user_name.txt') as f:
                lines = f.read()
                user_name = lines.split('\n', 1)[0]

        except FileNotFoundError:
            user_name = "jmniederle"


        run = wandb.init(project="product-sentiment", entity=user_name, config=config)

    else:
        run = None

    # Set device:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = "cpu"

    # Import GloVe Embeddings
    # Load training dataset to build vocab
    cache_path = os.path.join(get_project_root(), Path("sentiment_model/.vector_cache/"))
    glove_twitter = GloVe(name="twitter.27B", dim=embedding_dim, cache=cache_path)

    # Instantiate vectors and ensure a 0 vector is inserted for unknown characters and padded characters
    pre_embeds = glove_twitter.vectors
    pre_embeds = torch.cat((torch.zeros(2, pre_embeds.shape[1]), pre_embeds))

    # Load data:
    train_dataset = TweetDataset(split="train", dataset=dataset_name, pretrained_vecs=glove_twitter, subset=small_subset)
    valid_dataset = TweetDataset(split="valid", dataset=dataset_name, pretrained_vecs=glove_twitter, subset=small_subset)

    # Create data loaders:
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=pad_batch)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, collate_fn=pad_batch)

    if dataset_name == "sent_ex":
        num_classes = 3
        loss_fn = nn.CrossEntropyLoss()

    elif dataset_name == "sent140":
        num_classes = 1
        loss_fn = nn.BCEWithLogitsLoss()

    elif dataset_name == "sent140_multi_class":
        num_classes = 2
        loss_fn = nn.CrossEntropyLoss()

    # Initialize model
    model = SentimentNet(vocab_size=len(train_dataset.vocab), embedding_dim=embedding_dim,
                         rnn_hidden_dim=hidden_dim, rnn_n_layers=n_layers, rnn_bidirectional=bidirectional,
                         dropout_rate=dropout, num_classes=num_classes, pretrained_embeddings=pre_embeds,
                         freeze_embed=freeze_embed)

    # Initialize loss and optimizer

    optimizer_map = {
        "Adam": Adam,
        "SGD": SGD,
    }
    optimizer = optimizer_map[optimizer](model.parameters(), lr=lr)

    train(model, loss_fn, optimizer, train_loader, valid_loader,
          epochs=epochs, logging_freq=logging_freq, logging=run,
          device=device, checkpoint_path=checkpoint_path, save_checkpoint=save_checkpoint, num_classes=num_classes,
          config=config)


if __name__ == "__main__":
    run_training(save_checkpoint=False, small_subset=1000, wandb_logging=False)

# TODO: find out why gpu utilization is only 50% when running on gpu
