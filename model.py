"""Main container for common language model"""
import torch
import torch.nn as nn

from utils import get_mask

class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a criterion (decoder and loss function)."""

    def __init__(self, ntoken, ninp, nhid, nlayers, criterion, dropout=0.5):
        super(RNNModel, self).__init__()
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp)
        self.rnn = nn.LSTM(ninp, nhid, nlayers, dropout=dropout, batch_first=True)
        # Usually we use the same # dim in both input and output embedding
        self.proj = nn.Linear(nhid, ninp)

        self.nhid = nhid
        self.nlayers = nlayers
        self.criterion = criterion

        self.criterion.emb.weight = self.encoder.weight
        self.reset_parameters()

    def reset_parameters(self):
        init_range = 0.1
        self.encoder.weight.data.uniform_(-init_range, init_range)

    def _rnn(self, input):
        '''Serves as the encoder and recurrent layer'''
        emb = self.drop(self.encoder(input))
        output, unused_hidden = self.rnn(emb)
        output = self.proj(output)
        output = self.drop(output)
        return output

    def forward(self, input, target, length):

        mask = get_mask(length.data, max_len=input.size(1))
        rnn_output = self._rnn(input)
        loss, real_loss = self.criterion(target, rnn_output)
        loss = torch.masked_select(loss, mask)
        real_loss = torch.masked_select(real_loss, mask)

        return (loss.sum(), real_loss.sum())

    def forward_normalized(self, input, target, length):

        mask = get_mask(length.data, max_len=input.size(1))
        rnn_output = self._rnn(input)
        l1, l2 = self.criterion.forward_normalized(target, rnn_output)
        l1 = torch.masked_select(l1, mask)
        l2 = torch.masked_select(l2, mask)
        return l1, l2

    def forward_fast(self, input, target, length):

        mask = get_mask(length.data, max_len=input.size(1))
        rnn_output = self._rnn(input)
        l1 = self.criterion.forward_fast(target, rnn_output)
        return l1

    def forward_slow(self, input, target, length):

        mask = get_mask(length.data, max_len=input.size(1))
        rnn_output = self._rnn(input)
        l1 = self.criterion.forward_slow(target, rnn_output)
        l1 = torch.masked_select(l1, mask)
        return l1

    def loss_and_norm_term(self, input, target, length):

        mask = get_mask(length.data, max_len=input.size(1))
        rnn_output = self._rnn(input)
        loss = self.criterion(target, rnn_output)
        loss = torch.masked_select(loss, mask)

        return loss.sum()

    def forward_noreduce(self, input, target, length):

        mask = get_mask(length.data, max_len=input.size(1))
        rnn_output = self._rnn(input)
        loss, real_loss = self.criterion(target, rnn_output)
        loss = torch.masked_select(loss, mask)
        real_loss = torch.masked_select(real_loss, mask)

        return (loss, real_loss)
