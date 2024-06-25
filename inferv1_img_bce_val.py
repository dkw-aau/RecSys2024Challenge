#!/usr/bin/env python
# coding: utf-8


import polars as pl
import numpy as np
from tqdm.auto import tqdm
import torch
from data import load_parquets_from_zip, merge_article_with_imgs
from data.dataset import EkstraDataset, ekstra_inference_collate
from model import to_device, load_checkpoint
from model.model_v1 import EsktraSort, interpret_inference
import random
import pickle

print('Infering v1 imgs bin')

random.seed(42)
torch.manual_seed(42)
np.random.seed(42)

debug = False

print('Loading dataset...')

behaviors = pl.read_parquet('preprocess/test_behaviors.parquet')
history = pl.read_parquet('preprocess/test_history.parquet')
article = pl.read_parquet('preprocess/article.parquet')
images_embeddings = pl.read_parquet('preprocess/image_embs.parquet')
categories = pl.read_parquet('preprocess/categories_embs.parquet')

article_embeddings = load_parquets_from_zip('dataset/FacebookAI_xlm_roberta_base.zip')['FacebookAI_xlm_roberta_base/xlm_roberta_base'] 
art_img_embeddings = merge_article_with_imgs(article_embeddings, images_embeddings, col='embeddings')

filtered_behaviors = behaviors.filter(~pl.col('is_beyond_accuracy'))

if debug:
    filtered_behaviors = filtered_behaviors[:1_000]

ds = EkstraDataset(filtered_behaviors, history, article, art_img_embeddings, categories, labels=False)

del filtered_behaviors

dl = torch.utils.data.DataLoader(ds, batch_size=10, collate_fn=ekstra_inference_collate, shuffle=False)




model = EsktraSort()




if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'



path = 'checkpoint_v1_img_bce_val'
print('Loading model')
epoch = load_checkpoint(path, model)
print(epoch)




model = model.to(device)
model.eval()


res = []
i = 0
with torch.no_grad():
    for idx, (in_view_len, behavior), history in tqdm(dl):
        torch.cuda.empty_cache()
        behavior = to_device(behavior, device)
        history = to_device(history, device)
        
        pred = model(behavior, history)

        res.extend(interpret_inference(idx, pred.cpu().numpy(), in_view_len))
        i += 1
        if debug and i == 20:
            break


del ds
del dl


behaviors = pl.read_parquet('preprocess/test_behaviors.parquet')
history = pl.read_parquet('preprocess/test_history.parquet')
article = pl.read_parquet('preprocess/article.parquet')
images_embeddings = pl.read_parquet('preprocess/image_embs.parquet')
categories = pl.read_parquet('preprocess/categories_embs.parquet')

article_embeddings = load_parquets_from_zip('dataset/FacebookAI_xlm_roberta_base.zip')['FacebookAI_xlm_roberta_base/xlm_roberta_base'] 
art_img_embeddings = merge_article_with_imgs(article_embeddings, images_embeddings, col='embeddings')
filtered_behaviors = behaviors.filter(pl.col('is_beyond_accuracy'))

if debug:
    filtered_behaviors = filtered_behaviors[:1_000]

ds = EkstraDataset(filtered_behaviors, history, article, art_img_embeddings, categories, labels=False)



dl = torch.utils.data.DataLoader(ds, batch_size=10, collate_fn=ekstra_inference_collate, shuffle=False)



i = 0
with torch.no_grad():
    for idx, (in_view_len, behavior), history in tqdm(dl):
        torch.cuda.empty_cache()
        behavior = to_device(behavior, device)
        history = to_device(history, device)
        
        pred = model(behavior, history)
        
        res.extend(interpret_inference(idx, pred.cpu().numpy(), in_view_len))
        i += 1
        if debug and i == 20:
            break



with open(f'inference_v1_img_bce_val_epoch_{epoch}.pickle', 'wb') as f:
    pickle.dump(res, f)

print('Done.')

