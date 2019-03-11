import pandas as pd
import torch
from torch.utils.data import Dataset
from mecab import MeCab
from gluonnlp.data import PadSequence
from gluonnlp import Vocab

class Corpus(Dataset):
    """Corpus class"""
    def __init__(self, filepath: str, vocab: Vocab, tagger: MeCab, padder: PadSequence) -> None:
        """Instantiating Corpus class

        Args:
            filepath (str): filepath
            vocab (gluonnlp.Vocab): instance of gluonnlp.Vocab
            tagger (mecab.Mecab): instance of mecab.Mecab
            padder (gluonnlp.data.PadSequence): instance of gluonnlp.data.PadSequence
        """
        self.corpus = pd.read_table(filepath).loc[:, ['document', 'label']]
        self.vocab = vocab
        self.tagger = tagger
        self.padder = padder

    def __len__(self) -> int:
        return len(self.corpus)

    def __getitem__(self, idx) -> (torch.Tensor, torch.Tensor):
        tokenized = self.tagger.morphs(self.corpus.iloc[idx]['document'])
        tokenized2indices = torch.tensor(self.padder([self.vocab.token_to_idx[token] for token in tokenized]))
        labels = torch.tensor(self.corpus.iloc[idx]['label'])
        return tokenized2indices, labels








