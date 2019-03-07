import os, sys
import torch
import numpy as np
import gluonnlp as nlp
from model.data import Corpus
from model.net import SentenceCNN
from torch.utils.data import DataLoader
from mecab import MeCab
from tqdm import tqdm

# Restoring model
ckpt = torch.load('./checkpoint/model_ckpt.tar')
vocab = ckpt['vocab']
model = SentenceCNN(num_classes=2, vocab=vocab)
model.load_state_dict(ckpt['model_state_dict'])
model.eval()

# Creating Dataset, Dataloader
tagger = MeCab()
padder = nlp.data.PadSequence(length=30)
tst_filepath = os.path.join(os.getcwd(), 'data/preprocessed_test.txt')

tst_ds = Corpus(tst_filepath, vocab, tagger, padder)
tst_dl = DataLoader(tst_ds, batch_size=100, num_workers=4)

# Evaluation
correct_count = 0
for x_mb, y_mb in tqdm(tst_dl):
    with torch.no_grad():
        y_mb_hat = model(x_mb)
        y_mb_hat = torch.max(y_mb_hat, 1)[1].numpy()
        correct_count += np.sum((y_mb_hat) == y_mb.numpy())

print('Acc : {:.2%}'.format(correct_count / len(tst_ds)))