#!/usr/bin/env python
# coding: utf-8

import polars as pl
import numpy as np
from tqdm.auto import tqdm
import torch
from data import load_parquets_from_zip, merge_article_with_imgs
from data.dataset import EkstraDataset, ekstra_train_collate
from data.utils import standarize
from model import to_device, load_checkpoint, checkpoint
from model.model_v1 import EsktraSort, balance_bce_loss
from model.utils import set_lr
import random
import os

from tqdm.auto import tqdm


print('Traning Model v1 img bce val...')

random.seed(42)
torch.manual_seed(42)
np.random.seed(42)


print('Loading dataset...')

behaviors = pl.read_parquet('preprocess/validation_behaviors.parquet')
history = pl.read_parquet('preprocess/validation_history.parquet')
article = pl.read_parquet('preprocess/article.parquet')
images_embeddings = pl.read_parquet('preprocess/image_embs.parquet')
categories = pl.read_parquet('preprocess/categories_embs.parquet')

article_embeddings = load_parquets_from_zip('dataset/FacebookAI_xlm_roberta_base.zip')['FacebookAI_xlm_roberta_base/xlm_roberta_base'] 
art_img_embeddings = merge_article_with_imgs(article_embeddings, images_embeddings, col='embeddings')


ds = EkstraDataset(behaviors, history, article, art_img_embeddings, categories)

dl = torch.utils.data.DataLoader(ds, batch_size=64, collate_fn=ekstra_train_collate, shuffle=True)
print('Loading Model v1 img val...')


model = EsktraSort()



if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

print(f'Running on {torch.cuda.get_device_name()}')

model = model.to(device)


path = 'checkpoint_v1_img_bce'
if os.path.exists('checkpoint_v1_img_bce_val'):
    path = 'checkpoint_v1_img_bce_val'

optimizer = torch.optim.Adam(model.parameters())

epoch = load_checkpoint(path, model, optimizer) + 1

path = 'checkpoint_v1_img_bce_val'
print(f'Running epoch: {epoch}')
    
acc_loss = 0
acc_hit_rate = 0
from collections import deque
ma_loss = deque(maxlen=100)
mm_hit_rate = deque(maxlen=100)
loader = tqdm(dl)
for (_, (in_view_len, behavior), history), (clicked, scroll) in loader:
    optimizer.zero_grad()
    torch.cuda.empty_cache()
    behavior = to_device(behavior, device)
    history = to_device(history, device)

    pred = model(behavior, history)
    c_loss, hit_rate = balance_bce_loss(pred, in_view_len, clicked)
    c_loss.backward()
    optimizer.step()
    c_loss = c_loss.item()
    ma_loss.append(c_loss)
    mm_hit_rate.append(hit_rate)
    loader.set_postfix(hit_rate=hit_rate, loss=c_loss, ma_loss=np.mean(ma_loss), mm_hit_rate=np.mean(mm_hit_rate))
    acc_loss += c_loss
    acc_hit_rate += hit_rate
    
print(f'Current Loss: {acc_loss / len(dl)} {acc_hit_rate / len(dl)}')
checkpoint(path, model, epoch, acc_loss / len(dl), optimizer)


print('Done: Bye, bye!')




