from argparse import ArgumentParser
import sys

from pathlib import Path

import numpy as np
import wandb
from torch.utils.data import DataLoader
from torchtext.vocab import build_vocab_from_iterator, vectors, vocab, GloVe
from torch.optim import Adam, SGD
import torch.nn as nn
from train import train
import wandb as experiment_logger

import torch

from data_utils.tweet_dataset import TweetDataset, pad_batch
from model import SentimentNet


def run_evaluation(
        batch_size: int = 16,
        lr: float = 0.001,
        epochs: int = 15,
        embedding_dim: int = 50,
        hidden_dim: int = 256,
        n_layers: int = 2,
        bidirectional: bool = True,
        num_classes: int = 3,
        activation_fn: str = "relu",
        dropout: float = 0.5,
        optimizer: str = "Adam",
        freeze_embed: bool = False,
        logging_freq: int = 500,
        checkpoint_path: Path = Path("checkpoints/"),
        save_checkpoint: bool = False,
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

    # Set device:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    # Import GloVe Embeddings
    glove_twitter = GloVe(name="twitter.27B", dim=embedding_dim)

    # Instantiate vectors and ensure a 0 vector is inserted for unknown characters and padded characters
    pre_embeds = glove_twitter.vectors
    pre_embeds = torch.cat((torch.zeros(2, pre_embeds.shape[1]), pre_embeds))

    # Load data:
    test_dataset = TweetDataset(split="test", pretrained_vecs=glove_twitter)

    # Create data loaders:
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, collate_fn=pad_batch)

    # Load checkpoint:
    checkpoint = torch.load(checkpoint_path)

    # Initialize model
    model = SentimentNet(vocab_size=len(test_dataset.vocab), embedding_dim=embedding_dim,
                         rnn_hidden_dim=hidden_dim, rnn_n_layers=n_layers, rnn_bidirectional=bidirectional,
                         dropout_rate=dropout, num_classes=num_classes, pretrained_embeddings=pre_embeds,
                         freeze_embed=freeze_embed)


    model.to(device)



    train(model, loss_fn, optimizer, train_loader, valid_loader,
          epochs=epochs, logging_freq=logging_freq, logging=run,
          device=device, checkpoint_path=checkpoint_path, save_checkpoint=save_checkpoint)


run_training()