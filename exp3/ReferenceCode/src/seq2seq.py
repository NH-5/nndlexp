"""Seq2Seq model with GRU."""

import torch
import torch.nn as nn

from src.loss import NLLLoss


def gru_default_state(batch_size, input_size, hidden_size, num_layers=1, bidirectional=False):
    del input_size
    num_directions = 2 if bidirectional else 1
    return torch.zeros(num_layers * num_directions, batch_size, hidden_size)


class GRU(nn.Module):
    def __init__(self, config, is_training=True):
        super().__init__()
        self.batch_size = config.batch_size if is_training else config.eval_batch_size
        self.hidden_size = config.hidden_size
        self.rnn = nn.GRU(self.hidden_size, self.hidden_size)

    def forward(self, x, hidden):
        y1, h1 = self.rnn(x, hidden)
        return y1, h1

    def construct(self, x, hidden):
        return self.forward(x, hidden)


class Encoder(nn.Module):
    def __init__(self, config, is_training=True):
        super().__init__()
        self.vocab_size = config.en_vocab_size
        self.hidden_size = config.hidden_size
        self.batch_size = config.batch_size if is_training else config.eval_batch_size
        self.embedding = nn.Embedding(self.vocab_size, self.hidden_size)
        self.gru = GRU(config, is_training=is_training)

    def forward(self, encoder_input):
        embeddings = self.embedding(encoder_input)
        embeddings = embeddings.transpose(0, 1)
        h = torch.zeros(1, encoder_input.size(0), self.hidden_size, device=encoder_input.device)
        output, hidden = self.gru(embeddings, h)
        return output, hidden

    def construct(self, encoder_input):
        return self.forward(encoder_input)


class Decoder(nn.Module):
    def __init__(self, config, is_training=True):
        super().__init__()
        self.vocab_size = config.ch_vocab_size
        self.hidden_size = config.hidden_size
        self.embedding = nn.Embedding(self.vocab_size, self.hidden_size)
        self.gru = GRU(config, is_training=is_training)
        self.dense = nn.Linear(self.hidden_size, self.vocab_size)
        self.softmax = nn.LogSoftmax(dim=2)

    def forward(self, decoder_input, hidden):
        embeddings = self.embedding(decoder_input)
        embeddings = embeddings.transpose(0, 1)
        output, hidden = self.gru(embeddings, hidden)
        output = self.dense(output)
        output = self.softmax(output)
        return output, hidden

    def construct(self, decoder_input, hidden):
        return self.forward(decoder_input, hidden)


class Seq2Seq(nn.Module):
    def __init__(self, config, is_train=True):
        super().__init__()
        self.max_len = config.max_seq_length
        self.is_train = is_train
        self.encoder = Encoder(config, is_train)
        self.decoder = Decoder(config, is_train)

    def forward(self, src, dst):
        encoder_output, _ = self.encoder(src)
        decoder_hidden = encoder_output[self.max_len - 2 : self.max_len - 1, :, :]

        if self.is_train:
            outputs, _ = self.decoder(dst, decoder_hidden)
        else:
            decoder_input = dst[:, 0:1]
            decoder_outputs = []
            for _ in range(self.max_len):
                decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
                decoder_output = torch.argmax(decoder_output, dim=2).transpose(0, 1)
                decoder_outputs.append(decoder_output)
                decoder_input = decoder_output
            outputs = torch.cat(decoder_outputs, dim=1)
        return outputs

    def construct(self, src, dst):
        return self.forward(src, dst)


class WithLossCell(nn.Module):
    def __init__(self, backbone, config):
        super().__init__()
        self._backbone = backbone
        self.batch_size = config.batch_size
        self._loss_fn = NLLLoss()
        self.max_len = config.max_seq_length

    def forward(self, src, dst, label):
        out = self._backbone(src, dst)
        logits = out.transpose(0, 1).reshape(-1, out.size(-1))
        target = label.reshape(-1)
        return self._loss_fn(logits, target)

    def construct(self, src, dst, label):
        return self.forward(src, dst, label)


class InferCell(nn.Module):
    def __init__(self, network, config):
        super().__init__()
        del config
        self.network = network

    def forward(self, src, dst):
        return self.network(src, dst)

    def construct(self, src, dst):
        return self.forward(src, dst)
