import torch
import torch.nn.functional as F
import torch.nn as nn
from .utils import interpret_inference
import numpy as np


class ArticleEmbedding(nn.Module):

    def __init__(self, article_emb_size, cat_emb_size, premium, \
                  sentiments, temporal, weekdays, hours, dims, dropout):
        super().__init__()
        self.summarize = nn.Sequential(nn.Linear(article_emb_size + cat_emb_size, dims), 
                                       nn.SELU(),
                                       nn.Dropout(dropout),
                                       nn.Linear(dims, dims)
                                       ) 
        self.premium_embs = nn.Embedding(premium, dims)
        self.sentiment_embs = nn.Embedding(sentiments, dims)
        self.temporal_embs = nn.Embedding(temporal, dims)
        self.weekday_embs = nn.Embedding(weekdays, dims)
        self.hour_embs = nn.Embedding(hours, dims)
        self.dropout = dropout
        pass

    def forward(self, article, temporal, weekdays, hours):
        embs, cat_embs, premium, sentiment, mask = article
        x = self.summarize(torch.cat((embs, cat_embs), dim=-1))
        x += F.dropout(self.premium_embs(premium), self.dropout, self.training)
        x += F.dropout(self.sentiment_embs(sentiment), self.dropout, self.training)
        x += F.dropout(self.temporal_embs(temporal), self.dropout, self.training)
        x += F.dropout(self.weekday_embs(weekdays), self.dropout, self.training)
        x += F.dropout(self.hour_embs(hours), self.dropout, self.training)
        return x, mask
    

class EsktraSort(nn.Module):

    def __init__(self, device=4, sso=2, gender=4, postcode=6, 
                 age=12, subscriber=2, weekday=7, hour=24, premium=2, sentiment=3,
                 temporal=100, dims=32, txt_dims=768, img_dims=128, nhead=4, 
                 num_encoder_layers=3, num_decoder_layers=3, 
                 dim_feedforward=128, dropout=0.1):
        super().__init__()
        self.device_embs = nn.Embedding(device, dims)
        self.sso_embs = nn.Embedding(sso, dims)
        self.gender_embs = nn.Embedding(gender, dims)
        self.postcode_embs = nn.Embedding(postcode, dims)
        self.age_embs = nn.Embedding(age, dims)
        self.subscriber_embs = nn.Embedding(subscriber, dims)

        self.article_embs = ArticleEmbedding(txt_dims + img_dims, txt_dims, premium, sentiment,
                                             temporal, weekday, hour, dims, dropout)
        self.dropout = dropout

        self.transformer = nn.Transformer(d_model=dims, nhead=nhead,
                                          num_encoder_layers=num_encoder_layers,
                                          num_decoder_layers=num_decoder_layers,
                                          dim_feedforward=dim_feedforward,
                                          dropout=dropout, batch_first=True)
        
        self.sort_key = nn.Sequential(nn.Linear(2 * dims, dims), 
                                       nn.SELU(),
                                       nn.Dropout(dropout),
                                       nn.Linear(dims, 1)
                                       ) 
        pass

    def forward(self, behaviour, history):        
        history_emb, history_mask = self.article_embs(*history)
        articles, article_delta_time, impression_weekday, \
            impression_hour, device_type, is_sso_user, gender,\
            postcode, age, is_subscriber = behaviour
        
        articles_emb, articles_mask = self.article_embs(articles, article_delta_time,
                                                         impression_weekday, impression_hour)
        x = self.transformer(history_emb, articles_emb, 
                             src_key_padding_mask=history_mask, 
                             tgt_key_padding_mask=articles_mask,
                             memory_key_padding_mask=history_mask)
        
        b = F.dropout(self.device_embs(device_type), self.dropout, self.training)
        b += F.dropout(self.sso_embs(is_sso_user), self.dropout, self.training)
        b += F.dropout(self.gender_embs(gender), self.dropout, self.training)
        b += F.dropout(self.postcode_embs(postcode), self.dropout, self.training)
        b += F.dropout(self.age_embs(age), self.dropout, self.training)
        b += F.dropout(self.subscriber_embs(is_subscriber), self.dropout, self.training)
        b = b.repeat(1, x.shape[1], 1)

        x = torch.concat((x, b), dim=-1)
        
        order = self.sort_key(x)
        return order.squeeze(-1)




def balance_bce_loss(pred, in_view_len, clicked):
    pred = [p[:x] for p, x in zip(pred, in_view_len)]
    res = 0
    hit = 0
    for p, c in zip(pred, clicked):
        c = list(set(c))
        ok = p[c]
        if torch.argmax(p).item() in c:
            hit += 1
        not_ok = p[[x for x in range(p.shape[0]) if x not in c]]
        res += F.binary_cross_entropy_with_logits(ok, torch.ones_like(ok))
        res += F.binary_cross_entropy_with_logits(not_ok, torch.zeros_like(not_ok))
    return res / len(pred), hit / len(pred)


def balance_bce_scroll_loss(pred, in_view_len, clicked, scroll):
    pred = [p[:x] for p, x in zip(pred, in_view_len)]
    res = 0
    hit = 0
    total = 0
    for p, c, s in zip(pred, clicked, scroll):
        s = 0.5 + 0.5 * s
        total += s
        c = list(set(c))
        ok = p[c]
        if torch.argmax(p).item() in c:
            hit += 1
        not_ok = p[[x for x in range(p.shape[0]) if x not in c]]
        res += s * (F.binary_cross_entropy_with_logits(ok, torch.ones_like(ok)) + 
               F.binary_cross_entropy_with_logits(not_ok, torch.zeros_like(not_ok)))
    return res / len(pred), hit / len(pred)


def cce_scroll_loss(pred, in_view_len, clicked, scroll):
    pred = [p[:x] for p, x in zip(pred, in_view_len)]
    res = 0
    hit = 0
    total = 0
    for p, c, s in zip(pred, clicked, scroll):
        s = 0.5 + 0.5 * s
        total += s
        c = list(set(c))
        if torch.argmax(p).item() in c:
            hit += 1
        target = torch.zeros_like(p)
        target[c] = 1
        res += s * (F.cross_entropy(p, target))
    return res / len(pred), hit / len(pred)