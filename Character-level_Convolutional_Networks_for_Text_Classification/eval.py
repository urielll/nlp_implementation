import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from model.utils import JamoTokenizer
from model.data import Corpus
from model.net import CharCNN
from gluonnlp.data import PadSequence
from tqdm import tqdm

# restoring model
ckpt = torch.load('./checkpoint/model_ckpt.tar')
tokenizer = JamoTokenizer()
padder = PadSequence(300)

model = CharCNN(num_classes=2, embedding_dim=64, dic=tokenizer.token2idx)
model.load_state_dict(ckpt['model_state_dict'])
model.eval()

# creating dataset, dataloader
tst_filepath = os.path.join(os.getcwd(), 'data/preprocessed_train.txt')
tst_ds = Corpus(tst_filepath, tokenizer, padder)
tst_dl = DataLoader(tst_ds, batch_size=128, num_workers=4)

# evaluation
correct_count = 0
for x_mb, y_mb in tqdm(tst_dl):
    with torch.no_grad():
        y_mb_hat = model(x_mb)
        y_mb_hat = torch.max(y_mb_hat, 1)[1].numpy()
        correct_count += np.sum((y_mb_hat) == y_mb.numpy())

print('Acc : {:.2%}'.format(correct_count / len(tst_ds)))