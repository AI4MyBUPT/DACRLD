#!/usr/bin/env python3
# coding: utf-8
# @Author  : Xinhao Mei @CVSSP, University of Surrey
# @E-mail  : x.mei@surrey.ac.uk

import math
import torch
import torch.nn as nn
import numpy as np
from models.BERT_Config import MODELS



class BertEncoder(nn.Module):

    def __init__(self, config):
        super(BertEncoder, self).__init__()

        bert_type = config.bert_encoder.type
        dropout = config.training.dropout

        self.tokenizer = MODELS[bert_type][1].from_pretrained(bert_type)
        if 'clip' not in bert_type:
            self.bert_encoder = MODELS[bert_type][0].from_pretrained(bert_type,
                                                                     add_pooling_layer=False,
                                                                     hidden_dropout_prob=dropout,
                                                                     attention_probs_dropout_prob=dropout,
                                                                     output_hidden_states=False)
        else:
            self.bert_encoder = MODELS[bert_type][0].from_pretrained(bert_type)

        if config.training.freeze:
            for name, param in self.bert_encoder.named_parameters():
                param.requires_grad = False

    def forward(self, captions):
        # device = next(self.parameters()).device
        device = torch.device('cuda')
        tokenized = self.tokenizer(captions, add_special_tokens=True,
                                   padding=True, return_tensors='pt')
        input_ids = tokenized['input_ids'].to(device)
        attention_mask = tokenized['attention_mask'].to(device)
        output = self.bert_encoder(input_ids=input_ids,
                                   attention_mask=attention_mask)[0]

        cls = output[:, 0, :]
        return cls


class W2VEncoder(nn.Module):
    def __init__(self, config):
        raise NotImplementedError('W2VEncoder')
    
class BertEmbInputEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        bert_type = config.bert_encoder.type
        dropout = config.training.dropout
        if 'clip' not in bert_type and 'distilbert' not in bert_type:
            self.bert_encoder = MODELS[bert_type][0].from_pretrained(bert_type,
                                                                     add_pooling_layer=False,
                                                                     hidden_dropout_prob=dropout,
                                                                     attention_probs_dropout_prob=dropout,
                                                                     output_hidden_states=False)
        else:
            self.bert_encoder = MODELS[bert_type][0].from_pretrained(bert_type)

        if config.training.freeze:
            for name, param in self.bert_encoder.named_parameters():
                param.requires_grad = False
            
        self.cls_token_embedding = nn.Embedding(1, MODELS[bert_type][2])
        self.pre_mapping = nn.Linear(config.input_embedding_dim, MODELS[bert_type][2],)

    def forward(self, embeddings, attn_masks):
        device = embeddings.device
        batch_size = embeddings.shape[0]
        # map embedding dimensions
        embeddings = self.pre_mapping(embeddings)
        # append [CLS] embedding to embeddings, append 1 to attn_masks
        # self.cls_token_embedding.weight: 1 d
        # embedding: b 1 d
        # attn_masks: b 1
        embeddings = torch.cat((self.cls_token_embedding.weight.unsqueeze(0).repeat(batch_size,1,1),embeddings),dim=1).to(device)
        attn_masks = torch.cat((torch.ones((batch_size,1),device=device),attn_masks),dim=1).to(device)
        output = self.bert_encoder(inputs_embeds=embeddings,
                                   attention_mask=attn_masks)[0]
        cls = output[:, 0, :]
        return cls
