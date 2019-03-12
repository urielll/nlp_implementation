import os
import pickle
import json
import fire
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch import optim
from mecab import MeCab
from model.data import Corpus
from model.net import SentenceCNN
from gluonnlp.data import PadSequence
from tqdm import tqdm

def train(cfgpath):
    # parsing json
    with open(os.path.join(os.getcwd(), cfgpath)) as io:
        params = json.loads(io.read())

    with open(params['filepath'].get('vocab'), mode='rb') as io:
        vocab = pickle.load(io)

    # creating model
    model = SentenceCNN(num_classes=params['model'].get('num_classes'), vocab=vocab)

    # creating dataset, dataloader
    tagger = MeCab()
    padder = PadSequence(length=30)

    batch_size = params['training'].get('batch_size')
    tr_filepath = os.path.join(os.getcwd(), params['filepath'].get('tr'))
    val_filepath = os.path.join(os.getcwd(), params['filepath'].get('val'))

    tr_ds = Corpus(tr_filepath, vocab, tagger, padder)
    tr_dl = DataLoader(tr_ds, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)

    val_ds = Corpus(val_filepath, vocab, tagger, padder)
    val_dl = DataLoader(val_ds, batch_size=batch_size)

    # training
    loss_fn = nn.CrossEntropyLoss()
    opt = optim.Adam(params=model.parameters(), lr=params['training'].get('learning_rate'))

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)

    epochs = params['training'].get('epochs')

    for epoch in tqdm(range(epochs), desc='epochs'):

        avg_tr_loss = 0
        avg_val_loss = 0
        tr_step = 0
        val_step = 0

        model.train()
        for x_mb, y_mb in tqdm(tr_dl, desc='iters'):
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
        for x_mb, y_mb in tqdm(val_dl):
            x_mb = x_mb.to(device)
            y_mb = y_mb.to(device)

            with torch.no_grad():
                score = model(x_mb)
                val_loss = loss_fn(score, y_mb)
                avg_val_loss += val_loss.item()
                val_step += 1
        else:
            avg_val_loss /= val_step

        tqdm.write('epoch : {}, tr_loss : {:.3f}, val_loss : {:.3f}'.format(epoch + 1, avg_tr_loss, avg_val_loss))

    ckpt = {'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'opt_state_dict': opt.state_dict(),
            'vocab': vocab}

    savepath = os.path.join(os.getcwd(), params['filepath'].get('ckpt'))
    torch.save(ckpt, savepath)

if __name__ == '__main__':
    fire.Fire(train)