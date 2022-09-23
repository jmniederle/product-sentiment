import torch
import spacy
import numpy as np
from torchtext.datasets import AG_NEWS
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from torchtext import data

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# tweet_data = load_dataset("SetFit/tweet_sentiment_extraction")
# tweet_data_train = tweet_data['train']
# tweet_data_test = tweet_data['test']
#
# sample_text = 'NO MORE MSHS!!!!!!!! Gotta go to work...too tired'
#
# sample_tokenized = tokenizer(sample_text)
#
# vocab = build_vocab_from_iterator(yield_tokens(tweet_data_train))
# print(f"vocab size: {len(vocab)}")
# text_pipeline = lambda x: vocab(tokenizer(x))
# label_pipeline = lambda x: x


class TweetDataset(Dataset):
    """Tweet Dataset """

    def __init__(self, split="train"):
        """
        Create a dataset using the HuggingFace dataset tweet_sentiment_extraction.

        Text is tokenized and converted to a vocabulary index.

        Args:
            split: choose train or test split
        """
        tweet_data = load_dataset("SetFit/tweet_sentiment_extraction")
        self.split = split
        self.data = tweet_data[split]
        self.tokenizer = get_tokenizer("spacy", language="en_core_web_sm")
        self.vocab = self.build_vocab()

    def build_vocab(self):
        """
        Build a vocabulary from the data.
        Returns: vocabulary
        """
        def yield_tokens(data_iter):
            for t in data_iter['text']:
                yield self.tokenizer(t)

        vocab = build_vocab_from_iterator(yield_tokens(self.data))
        print(f"vocab size: {len(vocab)}")

        return vocab

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Returns the text and label for a given index.
        Args:
            idx: index of sample

        Returns: tokenized text and label

        """
        text_pipeline = lambda x: self.vocab(self.tokenizer(x))
        text = text_pipeline(self.data[idx]['text'])
        label = self.data[idx]['label']
        return text, label


tweet_dataset = TweetDataset()
print(next(iter(tweet_dataset)))
