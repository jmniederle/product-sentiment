import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch


class SentimentNet(nn.Module):
    def __init__(self,
                 vocab_size=34096,
                 embedding_dim=128,
                 rnn_hidden_dim=256,
                 rnn_n_layers=2,
                 rnn_bidirectional=True,
                 dropout_rate=0.5,
                 num_classes=3,
                 pad_idx=0,
                 pretrained_embeddings=None,
                 freeze_embed=True):
        super().__init__()

        if pretrained_embeddings is not None:
            self.embedding = nn.Embedding.from_pretrained(pretrained_embeddings, sparse=False, freeze=freeze_embed)
            self.embedding_dim = pretrained_embeddings.shape[1]
            assert self.embedding.weight.shape == pretrained_embeddings.shape

        else:
            self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
            self.embedding_dim = embedding_dim

        self.rnn = nn.LSTM(self.embedding_dim, rnn_hidden_dim, rnn_n_layers, bidirectional=rnn_bidirectional,
                           dropout=dropout_rate, batch_first=True)

        rnn_hidden_output_size = rnn_hidden_dim * 2 if rnn_bidirectional else rnn_hidden_dim

        self.mlp = nn.Sequential(
            nn.Linear(rnn_hidden_output_size, rnn_hidden_output_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(rnn_hidden_output_size, rnn_hidden_output_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(rnn_hidden_output_size, rnn_hidden_output_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(rnn_hidden_output_size, num_classes),
        )

        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, text, text_lengths):
        # text has shape [batch size, max sentence length in batch]
        embedded = self.dropout(self.embedding(text))  # shape: [batch size, max sent len, emb dim]

        # Packing embeddings, ensuring only non-padded elements are processed by rnn,
        # enforce_sorted is set to false so that sequences do not have to be sorted in length
        packed_embedded = pack_padded_sequence(embedded, text_lengths.cpu(), batch_first=True, enforce_sorted=False)

        packed_output, (hidden, cell) = self.rnn(packed_embedded)

        # Unpack and transform packed sequence to a tensor
        output, output_lengths = pad_packed_sequence(packed_output, batch_first=True)

        # shapes:
        # output -> [batch_size, max sent len, hidden_dim * num_directions (2)]
        # hidden -> [batch_size, num_layer * num_directions, hidden_dim]
        # cell -> [batch_size, num_layer * num_directions, hidden_dim]
        hidden, cell = torch.permute(hidden, (1, 0, 2)), torch.permute(cell, (1, 0, 2))

        # concat final forward and backward layer:
        final_hidden = self.dropout(torch.cat((hidden[:, -2, :], hidden[:, -1, :]), dim=1))
        # shape [batch_size, hidden_dim * num_directions]

        out = self.mlp(final_hidden)

        return out


# TODO: try using mean embedding for unknown tokens instead of zeros
# TODO: fix error when unfreezing embeddings
# TODO: maybe try fasttext

