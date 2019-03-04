import os
import pandas as pd
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
# dataset = pd.read_table(os.path.join(os.getcwd(), 'data/ratings_train.txt'))

class NSMC(Dataset):
    def __init__(self, filepath, vocab):
        self.corpus = pd.read_table(filepath).loc[:,['document', 'label']]

    def __len__(self):
        return len(self.corpus)

    def __getitem__(self, idx):
        return self.corpus.iloc[idx]

nsmc_ds = NSMC(os.path.join(os.getcwd(), 'data/ratings_train.txt'))
nsmc_dl = DataLoader(dataset=nsmc_ds,batch_size=2, shuffle=True)


for x_mb, y_mb in nsmc_dl:
    print(x_mb)