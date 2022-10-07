import torch
import spacy
import numpy as np
from torchtext.datasets import AG_NEWS
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader, random_split
from torchtext import data
from transformers import BertTokenizer
from sklearn.model_selection import train_test_split
from torch.nn.utils.rnn import pad_sequence

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TweetDataset(Dataset):
    """Tweet Dataset """

    def __init__(self, split="train", tokenizer=None):
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
        self.vocab = self.build_vocab()

    def get_split_data(self):

        if self.split == "test":
            return self.tweet_data['test']['text'], self.tweet_data['test']['label']

        elif (self.split == "valid") or (self.split == "train"):
            X_train, X_valid, y_train, y_valid = train_test_split(self.tweet_data['train']['text'],
                                                                  self.tweet_data['train']['label'],
                                                                  test_size=0.2, random_state=42)

            if self.split == "train":
                return X_train, y_train

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

        def yield_tokens(data_iter):
            for t in data_iter:
                yield self.tokenizer(t)

        vocab = build_vocab_from_iterator(yield_tokens(self.tweet_data['train']['text']))

        return vocab

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
            return self.vocab(self.tokenizer(x))

        text = text_pipeline(self.X[idx])
        label = self.y[idx]
        return torch.tensor(text), label


def pad_batch(tweet_batch):
    x, y = zip(*tweet_batch)
    x_lens = [len(x_i) for x_i in x]
    x_pad = pad_sequence(x, batch_first=True, padding_value=0)

    return x_pad, torch.tensor(y), torch.tensor(x_lens)


train_dataset = TweetDataset(split="train")

for i in range(len(train_dataset)):
    x_t, y = train_dataset[i]

    if ((x_t > 0).sum() == 0).item():
        print(train_dataset.X[i])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False, collate_fn=pad_batch)
for i, (x_pad_i, y_i, x_len_i) in enumerate(train_loader):
    if 0 in x_len_i:
        print(i)
        print(x_len_i)
        print((x_len_i == 0).nonzero())
        print(x_pad_i[(x_len_i == 0).nonzero()])

# batch_n = 516*32
# batch_idx = 27
# print(train_dataset[batch_n+batch_idx])
# print(train_dataset.X[batch_n+batch_idx])

