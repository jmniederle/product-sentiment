import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchtext.vocab import GloVe

from data_utils.metrics import accuracy
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
        dropout: float = 0.5,
        freeze_embed: bool = False,
        checkpoint_path: Path = Path("checkpoints/"),
        model_file="vivid-thunder-47/vivid-thunder-47-epoch-7.pth"
):

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
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=pad_batch)

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

    test_out = torch.tensor([]).to(device)

    # Validation
    model.eval()
    with torch.no_grad():
        for batch_idx, (data, target, text_lengths) in enumerate(test_loader):
            data, target, text_lengths = data.to(device), target.to(device), text_lengths.to(device)

            # Forward pass
            output = model(data, text_lengths)

            test_out = torch.cat((test_out, output))

    acc = accuracy(test_out, target)
    print(f"Test set accuracy: {acc}")

    return test_out.numpy()


if __name__ == "__main__":
    run_evaluation()
