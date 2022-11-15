import re

import torch
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator, vocab
from utils import get_project_root
import os
from pathlib import Path
from torchtext.vocab import GloVe

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TweetDataset(Dataset):
    """Tweet Dataset """

    def __init__(self, dataset="sent_ex", split="train", tokenizer=None, pretrained_vecs=None, subset=None):
        """
        Create a dataset using the HuggingFace dataset tweet_sentiment_extraction.

        Text is tokenized and converted to a vocabulary index.

        Args:
            split: choose train or test split
        """

        self.split = split
        self.dataset_name = dataset
        self.tweet_data = self.load_data()
        self.X, self.y = self.get_split_data()

        if subset is not None:
            self.X = self.X[:subset]
            self.y = self.y[:subset]

        self.filter_empty_strings()
        self.tokenizer = get_tokenizer("spacy", language="en_core_web_sm") if not tokenizer else tokenizer
        self.pretrained_vecs = pretrained_vecs
        self.vocab = self.build_vocab()

    def load_data(self):
        if self.dataset_name == "sent_ex":
            return load_dataset("SetFit/tweet_sentiment_extraction")

        elif self.dataset_name == "sent140" or self.dataset_name == "sent140_multi_class":
            return load_dataset("sentiment140")

    def get_split_data(self):

        if self.split == "test":
            if self.dataset_name == "sent_ex":
                return self.tweet_data['test']['text'], self.tweet_data['test']['label']

            elif self.dataset_name == "sent140" or self.dataset_name == "sent140_multi_class":
                return self.tweet_data['test']['text'], self.tweet_data['test']['sentiment']

        elif (self.split == "valid") or (self.split == "train"):
            if self.dataset_name == "sent_ex":
                X_train, X_valid, y_train, y_valid = train_test_split(self.tweet_data['train']['text'],
                                                                      self.tweet_data['train']['label'],
                                                                      test_size=0.2, random_state=42)
            elif self.dataset_name == "sent140" or self.dataset_name == "sent140_multi_class":
                X_train, X_valid, y_train, y_valid = train_test_split(self.tweet_data['train']['text'],
                                                                      self.tweet_data['train']['sentiment'],
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

    def text_pipeline(self, x):
        if self.pretrained_vecs is not None:
            return self.vocab(process_token_list(tokenize(x, tokenizer=self.tokenizer, lower=True)))

        else:
            return self.vocab(tokenize(x, tokenizer=self.tokenizer, lower=False))

    def label_pipeline(self, y):
        if self.dataset_name == "sent_ex":
            return int(y)

        elif self.dataset_name == "sent140":
            if y == 4:
                return 1

            elif y == 2:
                return 0.5

            elif y == 0:
                return 0

        elif self.dataset_name == "sent140_multi_class":
            if self.split == "train" or self.split == "valid":
                return [1 if (i == 1 and y == 4) or (i == 0 and y == 0) else 0 for i in range(2)]

            elif self.split == "test":
                return [1 if (i == 0 and y == 0) or (i == 1 and y == 2) or (i == 2 and y == 4) else 0 for i in range(3)]

    def __getitem__(self, idx):
        """
        Returns the text and label for a given index.
        Args:
            idx: index of sample

        Returns: tokenized text and label

        """

        text = self.text_pipeline(self.X[idx])
        label = self.label_pipeline(self.y[idx])
        return torch.tensor(text), label


class TweetDatasetInference(Dataset):
    """Tweet Dataset """

    def __init__(self, tweets, train_pipeline):
        """
        Create a dataset using the HuggingFace dataset tweet_sentiment_extraction.

        Text is tokenized and converted to a vocabulary index.

        Args:
            split: choose train or test split
        """
        self.X = tweets
        self.filter_empty_strings()
        self.text_pipeline = train_pipeline

    def filter_empty_strings(self):
        """
        Remove empty strings and their corresponding label from the dataset
        Returns:

        """
        for idx, text in enumerate(self.X):
            if len(text) == 0:
                del self.X[idx]

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        """
        Returns the text and label for a given index.
        Args:
            idx: index of sample

        Returns: tokenized text and label

        """

        text = self.text_pipeline(self.X[idx])
        return torch.tensor(text)


def pad_batch(tweet_batch):
    x, y = zip(*tweet_batch)
    x_lens = [len(x_i) for x_i in x]
    x_pad = pad_sequence(x, batch_first=True, padding_value=0)

    return x_pad, torch.tensor(y), torch.tensor(x_lens)


def pad_batch_inference(tweet_batch):
    x_lens = [len(x_i) for x_i in tweet_batch]
    x_pad = pad_sequence(tweet_batch, batch_first=True, padding_value=0)

    return x_pad, torch.tensor(x_lens)


def process_token(token):
    # Different regex parts for smiley faces
    eyes = r"[8:=;]"
    nose = r"['`\-]?"

    def re_sub(pattern, repl, text):
        return re.sub(pattern, repl, text)

    # Match url
    token = re_sub(r"https?:\/\/\S+\b|www\.(\w+\.)+\S*", "<url>", token)

    # Match user
    token = re_sub(r"@\w+", "<user>", token)

    # Match smileys
    token = re_sub(r"{}{}[)dD]+|[)dD]+{}{}".format(eyes, nose, nose, eyes), "<smile>", token)
    token = re_sub(r"{}{}p+".format(eyes, nose), "<lolface>", token)
    token = re_sub(r"{}{}\(+|\)+{}{}".format(eyes, nose, nose, eyes), "<sadface>", token)
    token = re_sub(r"{}{}[\/|l*]".format(eyes, nose), "<neutralface>", token)
    token = re_sub(r"<3", "<heart>", token)

    # Match number
    token = re_sub(r"[-+]?[.\d]*[\d]+[:,.\d]*", "<number>", token)

    # Match hashtag
    token = re_sub(r"#", "<hashtag>", token)
    return token


def tokenize(x, tokenizer=None, lower=True):
    tokenizer = get_tokenizer("spacy", language="en_core_web_sm") if not tokenizer else tokenizer

    if lower:
        return tokenizer(x.lower())
    else:
        return tokenizer(x)


def process_token_list(tokens):
    return [process_token(t) for t in tokens]

# TODO: verify dataset with and without pretrained GLove is working
# TODO: add sentiment140 to dataset and retrain model
# TODO: make Dutch sentiment model


if __name__ == "__main__":
    # Import GloVe Embeddings
    cache_path = os.path.join(get_project_root(), Path("sentiment_model/.vector_cache/"))
    glove_twitter = GloVe(name="twitter.27B", dim=50, cache=cache_path)

    train_dataset = TweetDataset(split="test", dataset="sent140_multi_class", pretrained_vecs=glove_twitter)
    print(train_dataset[0])