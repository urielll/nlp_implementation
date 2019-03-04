import os, sys
import pickle
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch import optim
from mecab import MeCab
from model.data import NSMC
from model.net import SentenceCNN
import gluonnlp as nlp

# Creating Dataset, DataLoader
tagger = MeCab()
padder = nlp.data.PadSequence(length=30)
with open('./data/vocab.pkl', mode='rb') as io:
    vocab = pickle.load(io)

tr_filepath = os.path.join(os.getcwd(), 'data/preprocessed_train.txt')
val_filepath = os.path.join(os.getcwd(), 'data/preprocessed_val.txt')

tr_ds = NSMC(tr_filepath, vocab, tagger, padder)
tr_dl = DataLoader(tr_ds, batch_size=100, shuffle=True, num_workers=4, drop_last=True)

val_ds = NSMC(val_filepath, vocab, tagger, padder)
val_dl = DataLoader(val_ds, batch_size=100)

# Creating model
model = SentenceCNN(num_classes=2, vocab=vocab)

def init_weights(layer):
    if isinstance(layer, nn.Conv1d):
        nn.init.kaiming_uniform_(layer.weight)
    elif isinstance(layer, nn.Linear):
        nn.init.xavier_normal_(layer.weight)
model.apply(init_weights)

# training
loss_fn = nn.CrossEntropyLoss()
opt = optim.Adam(params = model.parameters(), lr =1e-3)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)

epochs = 10

for epoch in range(epochs):

    avg_tr_loss = 0
    avg_val_loss = 0
    tr_step = 0
    val_step = 0

    model.train()

    for x_mb, y_mb in tr_dl:
        x_mb = x_mb.to(device)
        y_mb = y_mb.to(device)
        score = model(x_mb)

        opt.zero_grad()
        tr_loss = loss_fn(score, y_mb)
        reg_term = torch.norm(model.fc.weight, p=2)
        tr_loss.add_(.5 * reg_term)
        tr_loss.backward()
        opt.step()

        avg_tr_loss += tr_loss.item()
        tr_step += 1
    else:
        avg_tr_loss /= tr_step

    model.eval()
    for x_mb, y_mb in val_dl:
        x_mb = x_mb.to(device)
        y_mb = y_mb.to(device)

        with torch.no_grad():
            score = model(x_mb)
            val_loss = loss_fn(score, y_mb)
            avg_val_loss += val_loss.item()
            val_step += 1
    else:
        avg_val_loss /= val_step

    print('epoch : {}, tr_loss : {:.3f}, val_loss : {:.3f}'.format(epoch + 1, avg_tr_loss, avg_val_loss))


ckpt = {'epoch': epoch,
        'model_state_dict' : model.state_dict(),
        'opt_state_dict' : opt.state_dict(),
        'vocab' : vocab}

torch.save(ckpt, './checkpoint/trained.tar')