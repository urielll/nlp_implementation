import os
import pandas as pd
import gluonnlp as nlp
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from mecab import MeCab

class NSMC(Dataset):
    def __init__(self, filepath, vocab, tagger, padder):
        self.corpus = pd.read_table(filepath).loc[:,['document', 'label']]
        self.vocab = vocab
        self.tagger = tagger
        self.padder = padder

    def __len__(self):
        return len(self.corpus)

    def __getitem__(self, idx):
        tokenized = self.tagger.morphs(self.corpus.iloc[idx]['document'])
        tokenized2indices = torch.tensor(self.padder([self.vocab.token_to_idx[token] for token in tokenized]))
        labels = torch.tensor(self.corpus.iloc[idx]['label'])
        return tokenized2indices, labels







