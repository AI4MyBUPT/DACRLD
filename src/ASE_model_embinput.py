#!/usr/bin/env python3
# coding: utf-8
# partially adapted from https://github.com/XinhaoMei/DCASE2021_task6_v2.git by Xinhao Mei


import math
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from .AudioEncoderRet import Cnn10, ResNet38, Cnn14, TFEnc_FeatInput
from .TextEncoderRet import BertEncoder, W2VEncoder, BertEmbInputEncoder
from .BERT_Config import MODELS

def l2norm(X):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=1, keepdim=True).sqrt()
    X = torch.div(X, norm)
    return X

class AudioEnc(nn.Module):

    def __init__(self, config):
        super(AudioEnc, self).__init__()

        if config.cnn_encoder.model == 'Cnn10':
            self.audio_enc = Cnn10(config)
        elif config.cnn_encoder.model == 'ResNet38':
            self.audio_enc = ResNet38(config)
        elif config.cnn_encoder.model == 'Cnn14':
            self.audio_enc = Cnn14(config)
        elif config.cnn_encoder.model == 'Extracted+TF':
            self.audio_enc = TFEnc_FeatInput(config.cnn_encoder)
        else:
            raise NotImplementedError('No such audio encoder network.')

        if config.cnn_encoder.pretrained:
            # loading pretrained CNN weights
            pretrained_cnn = torch.load('pretrained_models/audio_encoder/{}.pth'.
                                        format(config.cnn_encoder.model))['model']
            dict_new = self.audio_enc.state_dict().copy()
            trained_list = [i for i in pretrained_cnn.keys()
                            if not ('fc' in i or i.startswith('spec') or i.startswith('logmel'))]
            for i in range(len(trained_list)):
                dict_new[trained_list[i]] = pretrained_cnn[trained_list[i]]
            self.audio_enc.load_state_dict(dict_new)
        if config.training.freeze:
            for name, param in self.audio_enc.named_parameters():
                param.requires_grad = False

    def forward(self, inputs, audio_feature_lengths=None):
        if audio_feature_lengths is None:
            audio_encoded = self.audio_enc(inputs)
        else:
            audio_feature_masks = torch.zeros((inputs.shape[0],inputs.shape[1]),dtype=torch.bool).to(inputs.device)
            for i in range(audio_feature_masks.shape[0]):
                audio_feature_masks[i,:audio_feature_lengths[i]]=True
            audio_encoded = self.audio_enc(inputs,audio_feature_masks)
        return audio_encoded


class ASE(nn.Module):

    def __init__(self, config):
        super(ASE, self).__init__()

        self.l2 = config.training.l2
        joint_embed = config.joint_embed

        self.audio_enc = AudioEnc(config)

        if config.cnn_encoder.model == 'Cnn10':
            self.audio_linear = nn.Sequential(
                nn.Linear(512, joint_embed),
                nn.ReLU(),
                nn.Linear(joint_embed, joint_embed)
            )
        elif config.cnn_encoder.model == 'ResNet38' or config.cnn_encoder.model == 'Cnn14':
            self.audio_linear = nn.Sequential(
                nn.Linear(2048, joint_embed * 2),
                nn.ReLU(),
                nn.Linear(joint_embed * 2, joint_embed)
            )
        elif config.cnn_encoder.model == 'Extracted+TF':
            self.tx_dim = config.cnn_encoder.get('TF_dim',768)
            self.audio_linear = nn.Sequential(
                nn.Linear(self.tx_dim, joint_embed * 2),
                nn.ReLU(),
                nn.Linear(joint_embed * 2, joint_embed)
            )

        # self.audio_gated_linear = nn.Linear(joint_embed, joint_embed)
        if config.text_encoder == 'bert':
            self.text_enc = BertEncoder(config)
            bert_type = config.bert_encoder.type
            self.text_linear = nn.Sequential(
                nn.Linear(MODELS[bert_type][2], joint_embed * 2),
                nn.ReLU(),
                nn.Linear(joint_embed * 2, joint_embed)
            )
        elif config.text_encoder == 'w2v':
            self.text_enc = W2VEncoder(config)
            self.text_linear = nn.Sequential(
                nn.Linear(300, joint_embed),
                nn.ReLU(),
                nn.Linear(joint_embed, joint_embed)
            )
        elif config.text_encoder == 'bert_embinput':
            self.text_enc = BertEmbInputEncoder(config)
            bert_type = config.bert_encoder.type
            self.text_linear = nn.Sequential(
                nn.Linear(MODELS[bert_type][2], joint_embed * 2),
                nn.ReLU(),
                nn.Linear(joint_embed * 2, joint_embed)
            )

    def encode_audio(self, audios, audio_feature_lengths=None):
        if audio_feature_lengths == None:
            return self.audio_enc(audios)
        else:
            return self.audio_enc(audios, audio_feature_lengths)

    def encode_text(self, input_embeddings, input_attn_masks):
        return self.text_enc(input_embeddings, input_attn_masks)

    def forward(self, audios, input_embeddings, input_attn_masks, audio_feature_lengths=None):

        audio_encoded = self.encode_audio(audios, audio_feature_lengths)     # batch x channel
        caption_encoded = self.encode_text(input_embeddings, input_attn_masks)

        audio_embed = self.audio_linear(audio_encoded)

        caption_embed = self.text_linear(caption_encoded)

        if self.l2:
            # apply l2-norm on the embeddings
            audio_embed = l2norm(audio_embed)
            caption_embed = l2norm(caption_embed)

        return audio_embed, caption_embed

    def langevin_loss(self, audios, input_embeddings, input_attn_masks, audio_feature_lengths=None):
        audio_embed, caption_embed = self.forward(audios, input_embeddings, input_attn_masks, audio_feature_lengths)
        a_norm = torch.nn.functional.normalize(audio_embed, p=2, dim=1)
        c_norm = torch.nn.functional.normalize(caption_embed, p=2, dim=1)
        sim = torch.sum(a_norm*c_norm)
        sim_loss = -sim
        return {'loss':sim_loss,}
