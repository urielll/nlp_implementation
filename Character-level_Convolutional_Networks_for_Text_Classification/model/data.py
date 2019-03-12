import pandas as pd
import torch
from torch.utils.data import Dataset
from gluonnlp.data import PadSequence
from model.utils import JamoTokenizer

class Corpus(Dataset):
    """Corpus class"""
    def __init__(self, filepath: str, tokenizer: JamoTokenizer, padder: PadSequence) -> None:
        """Instantiating Corpus class

        Args:
            filepath (str): filepath
            padder (gluonnlp.data.PadSequence): instance of gluonnlp.data.PadSequence
        """
        self.corpus = pd.read_table(filepath).loc[:, ['document', 'label']]
        self.padder = padder
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        return len(self.corpus)

    def __getitem__(self, idx) -> (torch.Tensor, torch.Tensor):
        tokenized2indices = self.tokenizer.tokenize_and_transform(self.corpus.iloc[idx]['document'])
        tokenized2indices = torch.tensor(self.padder(tokenized2indices))
        labels = torch.tensor(self.corpus.iloc[idx]['label'])
        return tokenized2indices, labels
