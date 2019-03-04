import pandas as pd
import torch
import torch.nn as nn
from supervised.model import MorphConv
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from torch import optim
from mecab import MeCab
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import gluonnlp as nlp
import itertools

# loading and preprocessing dataset
dataset = pd.read_table('sample_data/ratings_train.txt').loc[:,['document','label']]
dataset = dataset.loc[dataset['document'].isna().apply(lambda elm : not elm), :]
tr, val = train_test_split(dataset, test_size=.2)

# preprocessing and creating the vocab
mecab_tagger = MeCab()
tr['document'] = tr['document'].apply(mecab_tagger.morphs)
val['document'] = val['document'].apply(mecab_tagger.morphs)
counter = nlp.data.count_tokens(itertools.chain.from_iterable([token for token in tr['document']]))
vocab = nlp.Vocab(counter=counter, min_freq=10, bos_token=None, eos_token=None)

## loading SISG embedding
ptr_embedding = nlp.embedding.create('fasttext', source='wiki.ko', load_ngrams=True)
vocab.set_embedding(ptr_embedding)

## token2idx
pad_sequence = nlp.data.PadSequence(length=30, pad_val=0)

x_tr = tr['document'].apply(lambda sen : pad_sequence([vocab.token_to_idx[token] for token in sen])).tolist()
x_tr = torch.tensor(x_tr)
y_tr = torch.tensor(tr['label'].tolist())

x_val = val['document'].apply(lambda sen : pad_sequence([vocab.token_to_idx[token] for token in sen])).tolist()
x_val = torch.tensor(x_val)
y_val = torch.tensor(val['label'].tolist())

# Dataset, DataLoader
## training
tr_dataset = TensorDataset(x_tr, y_tr)
tr_dataloader = DataLoader(dataset=tr_dataset, batch_size=100,
                           drop_last=True, shuffle=True)

val_dataset = TensorDataset(x_val, y_val)
val_dataloader = DataLoader(dataset=val_dataset, batch_size=100)

# Model
model = MorphConv(num_classes=2, vocab=vocab)

def init_weights(layer):
    if isinstance(layer, nn.Conv1d):
        nn.init.kaiming_uniform_(layer.weight)
    elif isinstance(layer, nn.Linear):
        nn.init.xavier_normal_(layer.weight)
model.apply(init_weights)

# training
loss_fn = nn.CrossEntropyLoss()

epochs = 10
opt = optim.Adam(params = model.parameters(), lr =1e-3)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)

for epoch in range(10):

    avg_tr_loss = 0
    avg_val_loss = 0
    tr_step = 0
    val_step = 0

    model.train()

    for x_mb, y_mb in tr_dataloader:
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
    for x_mb, y_mb in val_dataloader:
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

torch.save(ckpt, 'saved_model/trained.tar')