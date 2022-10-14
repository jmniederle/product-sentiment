import torch
import spacy
import numpy as np
from torchtext.datasets import AG_NEWS
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator, vectors, vocab, GloVe
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader, random_split
from torchtext import data
from transformers import BertTokenizer
from sklearn.model_selection import train_test_split
from torch.nn.utils.rnn import pad_sequence

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TweetDataset(Dataset):
    """Tweet Dataset """

    def __init__(self, split="train", tokenizer=None, pretrained_vecs=None):
        """
        Create a dataset using the HuggingFace dataset tweet_sentiment_extraction.

        Text is tokenized and converted to a vocabulary index.

        Args:
            split: choose train or test split
        """

        self.split = split
        self.tweet_data = load_dataset("SetFit/tweet_sentiment_extraction")
        self.X, self.y = self.get_split_data()
        self.filter_empty_strings()
        self.tokenizer = get_tokenizer("spacy", language="en_core_web_sm") if not tokenizer else tokenizer
        self.pretrained_vecs = pretrained_vecs
        self.vocab = self.build_vocab()

    def get_split_data(self):

        if self.split == "test":
            return self.tweet_data['test']['text'], self.tweet_data['test']['label']

        elif (self.split == "valid") or (self.split == "train"):
            X_train, X_valid, y_train, y_valid = train_test_split(self.tweet_data['train']['text'],
                                                                  self.tweet_data['train']['label'],
                                                                  test_size=0.2, random_state=42)

            if self.split == "train":
                return X_train[:500], y_train[:500]

            else:
                return X_valid, y_valid

        else:
            raise ValueError(f"Invalid split {self.split}")

    def filter_empty_strings(self):
        """
        Remove empty strings and their corresponding label from the dataset
        Returns:

        """
        for idx, text in enumerate(self.X):
            if len(text) == 0:
                del self.X[idx]
                del self.y[idx]

    def build_vocab(self):
        """
        Build a vocabulary from the data.
        Returns: vocabulary
        """

        if self.pretrained_vecs is not None:
            pretrained_vocab = vocab(self.pretrained_vecs.stoi, min_freq=0)

            # insert and set a token and idx for unknown characters
            pad_index = 0
            pad_token = "<pad>"
            pretrained_vocab.insert_token(pad_token, pad_index)
            unk_token = "<unk>"
            unk_index = 1
            pretrained_vocab.insert_token(unk_token, unk_index)
            pretrained_vocab.set_default_index(unk_index)

            return pretrained_vocab

        else:
            def yield_tokens(data_iter):
                for t in data_iter:
                    yield self.tokenizer(t)

            built_vocab = build_vocab_from_iterator(yield_tokens(self.tweet_data['train']['text']))

            return built_vocab

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        """
        Returns the text and label for a given index.
        Args:
            idx: index of sample

        Returns: tokenized text and label

        """

        def text_pipeline(x):
            if self.pretrained_vecs is not None:
                return self.vocab(self.tokenizer(x.lower()))

            else:
                return self.vocab(self.tokenizer(x))

        text = text_pipeline(self.X[idx])
        label = self.y[idx]
        return torch.tensor(text), label


def pad_batch(tweet_batch):
    x, y = zip(*tweet_batch)
    x_lens = [len(x_i) for x_i in x]
    x_pad = pad_sequence(x, batch_first=True, padding_value=0)

    return x_pad, torch.tensor(y), torch.tensor(x_lens)

# TODO: verify dataset with and without pretrained GLove is working
