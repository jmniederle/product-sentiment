from argparse import ArgumentParser
import sys

from pathlib import Path

import numpy as np
import wandb
from torch.utils.data import DataLoader

from torch.optim import Adam, SGD
import torch.nn as nn
from train import train
import wandb as experiment_logger

import torch

from data_utils.tweet_dataset import TweetDataset, pad_batch
from model import SentimentNet


def run_training(
        batch_size: int = 64,
        lr: float = 0.001,
        epochs: int = 15,
        embedding_dim: int = 128,
        hidden_dim: int = 256,
        n_layers: int = 2,
        bidirectional: bool = True,
        num_classes: int = 3,
        activation_fn: str = "relu",
        dropout: float = 0.3,
        optimizer: str = "Adam",
        gui: bool = False,
        logging_freq: int = 500,
        checkpoint_path: Path = Path("checkpoints/"),
        data_path: Path = Path("./data.pkl")
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
              "dropout": dropout}

    run = wandb.init(project="product-sentiment", entity="jmniederle", config=config)

    # Set device:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data:
    train_dataset = TweetDataset(split="train")
    valid_dataset = TweetDataset(split="valid")

    # Create data loaders:
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=pad_batch)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, collate_fn=pad_batch)

    # Initialize model
    model = SentimentNet(vocab_size=len(train_dataset.vocab), embedding_dim=embedding_dim,
                                 rnn_hidden_dim=hidden_dim, rnn_n_layers=n_layers, rnn_bidirectional=bidirectional,
                                 dropout_rate=dropout, num_classes=num_classes)

    # Initialize loss and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer_map = {
        "Adam": Adam,
        "SGD": SGD,
    }
    optimizer = optimizer_map[optimizer](model.parameters(), lr=lr)

    # batch_i = next(iter(train_loader))
    # batch_i_out = sentiment_net(batch_i[0], batch_i[2])
    # print(batch_i_out)

    train(model, loss_fn, optimizer, train_loader, valid_loader,
          epochs=epochs, logging_freq=logging_freq, logging=run,
          device=device, checkpoint_path=checkpoint_path)


run_training()
