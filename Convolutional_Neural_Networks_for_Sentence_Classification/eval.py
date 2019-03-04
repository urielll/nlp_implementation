import torch
import gluonnlp as nlp
import pandas as pd
import numpy as np
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from supervised.model import MorphConv
from mecab import MeCab

ckpt = torch.load('saved_model/trained.tar')
vocab = ckpt['vocab']
model = MorphConv(num_classes=2, vocab=vocab)
model.load_state_dict(ckpt['model_state_dict'])
model.eval()

# preparing tst_dataset
mecab_tagger = MeCab()
tst = pd.read_table('sample_data/ratings_test.txt').loc[:,['document','label']]
tst = tst.loc[tst['document'].isna().apply(lambda elm : not elm), :]
tst['document'] = tst['document'].apply(mecab_tagger.morphs)

pad_sequence = nlp.data.PadSequence(length=30, pad_val=0)
x_tst = tst['document'].apply(lambda sen : pad_sequence([vocab.token_to_idx[token] for token in sen])).tolist()
x_tst = torch.tensor(x_tst)
y_tst = torch.tensor(tst['label'].tolist())

# dataloader
tst_dataset = TensorDataset(x_tst, y_tst)
tst_dataloader = DataLoader(dataset=tst_dataset, batch_size=100, num_workers=4)

# evaluation
results = np.array([])
for x_mb, y_mb in tst_dataloader:
    with torch.no_grad():
        y_mb_hat = model(x_mb)
        y_mb_hat = torch.max(y_mb_hat, 1)[1].numpy()
        results = np.append(results, y_mb_hat)

print('Acc : {:.2%}'.format(np.mean(results == y_tst.numpy())))


