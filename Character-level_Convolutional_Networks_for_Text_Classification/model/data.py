import pandas as pd
import torch
from torch.utils.data import Dataset
from gluonnlp import Vocab
from gluonnlp.data import PadSequence
from mecab import MeCab
from model.utils import JamoTokenizer

class Corpus(Dataset):
    def __init__(self, filepath: str, tokenizer: JamoTokenizer, padder: PadSequence) -> None:
        """Instantiating CorpusForJamoCNN

        Args:
            filepath: filepath
            padder: instance of gluonnlp.data.PadSequence
        """
        self.corpus = pd.read_table(filepath).loc[:,['document', 'label']]
        self.padder = padder
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.corpus)

    def __getitem__(self, idx):
        tokenized2indices = self.tokenizer.tokenize_and_transform(self.corpus.iloc[idx]['document'])
        tokenized2indices = torch.tensor(self.padder(tokenized2indices))
        labels = torch.tensor(self.corpus.iloc[idx]['label'])
        return tokenized2indices, labels
