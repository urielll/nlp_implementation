import os
import json
import fire
import torch
from torch.utils.data import DataLoader
from model.utils import JamoTokenizer
from model.data import Corpus
from model.net import CharCNN
from gluonnlp.data import PadSequence
from tqdm import tqdm

def evaluate(cfgpath):
    # parsing json
    with open(os.path.join(os.getcwd(), cfgpath)) as io:
        params = json.loads(io.read())

    # restoring model
    savepath = os.path.join(os.getcwd(), params['filepath'].get('ckpt'))
    ckpt = torch.load(savepath)
    tokenizer = JamoTokenizer()
    padder = PadSequence(300)

    model = CharCNN(num_classes=params['model'].get('num_classes'),
                    embedding_dim=params['model'].get('embedding_dim'),
                    dic=tokenizer.token2idx)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)

    # creating dataset, dataloader
    tst_filepath = os.path.join(os.getcwd(), params['filepath'].get('tst'))
    tst_ds = Corpus(tst_filepath, tokenizer, padder)
    tst_dl = DataLoader(tst_ds, batch_size=128, num_workers=4)

    # evaluation
    correct_count = 0
    for x_mb, y_mb in tqdm(tst_dl):
        x_mb = x_mb.to(device)
        y_mb = y_mb.to(device)
        with torch.no_grad():
            y_mb_hat = model(x_mb)
            y_mb_hat = torch.max(y_mb_hat, 1)[1]
            correct_count += (y_mb_hat == y_mb).sum().item()

    print('Acc : {:.2%}'.format(correct_count / len(tst_ds)))

if __name__ == '__main__':
    fire.Fire(evaluate)