import logging

import torch


class Embedder(torch.nn.Module):
    def __init__(self, vocab, config):
        super().__init__()
        self.init_vocab(vocab, config['optimize_embeddings'])
        logging.info(f"Optimize embeddings = {config['optimize_embeddings']}")
        logging.info(f"Vocabulary size = {len(vocab.vectors)}")

    def init_vocab(self, vocab, optimize_embeddings=False, device=None):
        self.embedding_dim = vocab.vectors.shape[1]
        self.embeddings = torch.nn.Embedding(len(vocab), self.embedding_dim)
        self.embeddings.weight.data.copy_(vocab.vectors)
        self.embeddings.weight.requires_grad = optimize_embeddings
        self.vocab = vocab
        if device is not None:
            self.embeddings = self.embeddings.to(device)

    def forward(self, input):
        return self.embeddings(input)
