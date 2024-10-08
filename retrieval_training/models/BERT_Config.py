#!/usr/bin/env python3
# coding: utf-8
# @Author  : Xinhao Mei @CVSSP, University of Surrey
# @E-mail  : x.mei@surrey.ac.uk

from transformers import BertModel, BertTokenizer, GPT2Model, GPT2Tokenizer,\
    RobertaModel, RobertaTokenizer, DistilBertModel, DistilBertTokenizer,\
    CLIPTokenizer, CLIPTextModel

MODELS = {
    'openai/clip-vit-base-patch32': (CLIPTextModel, CLIPTokenizer, 512),
    'prajjwal1/bert-tiny': (BertModel, BertTokenizer, 128),
    'prajjwal1/bert-mini': (BertModel, BertTokenizer, 256),
    'prajjwal1/bert-small': (BertModel, BertTokenizer, 512),
    'prajjwal1/bert-medium': (BertModel, BertTokenizer, 512),
    'gpt2': (GPT2Model, GPT2Tokenizer, 768),
    'distilgpt2': (GPT2Model, GPT2Tokenizer, 768),
    'bert-base-uncased': (BertModel, BertTokenizer, 768),
    'pretrained_models/bert-base-uncased': (BertModel, BertTokenizer, 768),
    'bert-large-uncased': (BertModel, BertTokenizer, 1024),
    'roberta-base': (RobertaModel, RobertaTokenizer, 768),
    'roberta-large': (RobertaModel, RobertaTokenizer, 1024),
    'distilbert-base-uncased': (DistilBertModel, DistilBertTokenizer, 768),
    'pretrained_models/distilbert-base-uncased': (DistilBertModel, DistilBertTokenizer, 768),
    "distilroberta-base": (RobertaModel, RobertaTokenizer, 768),
}
