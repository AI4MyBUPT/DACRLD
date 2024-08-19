import math
import copy
from pathlib import Path
from random import random
from functools import partial
from collections import namedtuple
from multiprocessing import cpu_count
import os
from mmcv.utils.config import ConfigDict

import torch
from torch import nn, einsum
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from torch.optim import AdamW
from torchvision import transforms as T, utils

from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange

from torch.nn import TransformerEncoder, TransformerEncoderLayer, \
    TransformerDecoder, TransformerDecoderLayer

from .x_transformer import AbsolutePositionalEmbedding, Encoder, group_dict_by_key, string_begins_with
from .Encoder import Cnn10, Cnn14, ResNet38
# partially adapted from https://github.com/lucidrains/denoising-diffusion-pytorch and https://github.com/justinlovelace/latent-diffusion-for-language

class MLPLayers(nn.Module):
    def __init__(self, units=[512, 512, 512], nonlin:nn.Module=nn.ReLU(), dropout=0.1):
        super(MLPLayers, self).__init__()
        self.nonlin = nonlin
        self.dropout = dropout

        sequence = []
        for u0, u1 in zip(units[:-1], units[1:]):
            sequence.append(nn.Linear(u0, u1))
            sequence.append(self.nonlin)
            sequence.append(nn.Dropout(self.dropout))
        sequence = sequence[:-2]

        self.sequential = nn.Sequential(*sequence)

    def forward(self, X):
        X = self.sequential(X)
        return X


# Helper Functions
def groupby_prefix_and_trim(prefix, d):
    kwargs_with_prefix, kwargs = group_dict_by_key(partial(string_begins_with, prefix), d)
    kwargs_without_prefix = dict(map(lambda x: (x[0][len(prefix):], x[1]), tuple(kwargs_with_prefix.items())))
    return kwargs_without_prefix, kwargs

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def exists(val):
    return val is not None

def init_zero_(layer):
    nn.init.constant_(layer.weight, 0.)
    if exists(layer.bias):
        nn.init.constant_(layer.bias, 0.)

# sinusoidal positional embeds


class PositionalEncodingM(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=2000):
        super(PositionalEncodingM, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional audio_encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        """

        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * (-emb))
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class DiffusionLatentTransformer(nn.Module):
    def __init__(
        self,
        config: ConfigDict,
    ):
        super().__init__()

        self.model_type = 'CNN+DiffusionLatentTransformer'
        self.config = config
        tx_dim = config.tx_dim
        self.tx_dim = tx_dim
        tx_depth = config.tx_depth
        heads = config.tx_heads
        latent_dim = config.get('latent_dim', None)
        max_seq_len = config.get('max_seq_len', 64)
        self_condition = config.get('self_condition', False)
        dropout = config.get('dropout', 0.1)
        scale_shift = config.get('scale_shift', False)
        class_conditional = config.get('class_conditional', False)
        num_classes = config.get('num_classes', 0)
        class_unconditional_prob = config.get('class_unconditional_prob', 0)
        # setting for CNN
        self.audio_feat_extract_dim = config.get('audio_feat_extract_dim', 1024)
        self.audio_linear = nn.Linear(self.audio_feat_extract_dim, self.tx_dim, bias=True)
        if config.src_encoder.model == 'Cnn10':
            self.feature_extractor = Cnn10(config)
        elif config.src_encoder.model == 'Cnn14':
            self.feature_extractor = Cnn14(config)
        elif config.src_encoder.model == 'ResNet38':
            self.feature_extractor = ResNet38(config)
        elif config.src_encoder.model == 'Extracted':
            self.feature_extractor = nn.Identity()
            assert config.src_encoder.pretrained == False, 'Extracted feature cannot load model'
            assert config.src_encoder.frozen == False, 'Extracted feature have no model, cannot be frozen'
            config.src_encoder.pretrained = False
            config.src_encoder.frozen = False
        else:
            raise NameError('No such encoder model')

        if config.src_encoder.pretrained:
            pretrained_cnn = torch.load('pretrained_models/audio_encoder/{}.pth'.
                                        format(config.src_encoder.model),map_location='cpu')['model']
            dict_new = self.feature_extractor.state_dict().copy()
            trained_list = [i for i in pretrained_cnn.keys()
                            if not ('fc' in i or i.startswith('spec') or i.startswith('logmel'))]
            for i in range(len(trained_list)):
                dict_new[trained_list[i]] = pretrained_cnn[trained_list[i]]
            self.feature_extractor.load_state_dict(dict_new)
        if config.src_encoder.frozen: # name 'freeze' conflict with dict interior function
            print('Freezing audio_encoder')
            for name, p in self.feature_extractor.named_parameters():
                if 'fc' not in name:
                    p.requires_grad = False
        else:
            print('Not freezing audio_encoder')

        self.latent_dim = latent_dim

        self.self_condition = self_condition
        self.scale_shift = scale_shift
        self.class_conditional = class_conditional
        self.num_classes = num_classes
        self.class_unconditional_prob = class_unconditional_prob

        self.max_seq_len = max_seq_len

        # time embeddings
        sinu_pos_emb = SinusoidalPosEmb(tx_dim)

        time_emb_dim = tx_dim*4
        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(tx_dim, time_emb_dim),
            nn.GELU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )
        self.time_pos_embed_mlp = nn.Sequential(
                nn.GELU(),
                nn.Linear(time_emb_dim, tx_dim)
            )

        self.pos_emb = AbsolutePositionalEmbedding(tx_dim, max_seq_len)

        self.cross = True  # *audio_conditional* always, self_condition, class_conditional

        self.encoder = Encoder(
            dim=tx_dim,
            depth=tx_depth,
            heads=heads,
            attn_dropout=dropout,    # dropout post-attention
            ff_dropout=dropout,       # feedforward dropout
            rel_pos_bias=True,
            ff_glu=True,
            cross_attend=self.cross,
            time_emb_dim=tx_dim*4 if self.scale_shift else None,
        )
        if self_condition:
            self.null_embedding = nn.Embedding(1, tx_dim)
            self.context_proj = nn.Linear(latent_dim, tx_dim)
            
        if self.class_conditional:
            assert num_classes > 0
            self.class_embedding = nn.Embedding(num_classes+1, tx_dim)

        self.input_proj = nn.Linear(latent_dim, tx_dim)
        self.norm = nn.LayerNorm(tx_dim)
        self.output_proj = nn.Linear(tx_dim, latent_dim)

        # additional positional embedding
        self.req_audio_pos_emb = config.get('req_audio_pos_emb',False)
        if self.req_audio_pos_emb:
            self.audio_pos_emb = AbsolutePositionalEmbedding(tx_dim, max_seq_len=2000)
        self.req_self_condition_pos_emb = config.get('req_self_condition_pos_emb',False)
        # add token_type embedding for audio, self-condition, keyword-condition, and reference-condition
        if self.self_condition:
            self.self_condition_token_type_embedding = nn.Embedding(1, tx_dim)
        self.keyword_condition = config.get('keyword_condition',False)
        self.keyword_condition_null_prob = config.get('keyword_condition_null_prob',0)
        if self.keyword_condition:
            self.keyword_condition_token_type_embedding = nn.Embedding(1, tx_dim)
        self.reference_condition = config.get('reference_condition',False)
        if self.reference_condition:
            self.reference_condition_token_type_embedding = nn.Embedding(1, tx_dim)
        self.req_audio_token_type_emb = config.get('req_audio_token_type_emb',None)
        if self.req_audio_token_type_emb is None:
            self.req_audio_token_type_emb = self.self_condition or self.keyword_condition or self.reference_condition
        if self.req_audio_token_type_emb:
            self.audio_token_type_embedding = nn.Embedding(1, tx_dim)
        # add projection and null_embedding for keyword_condition
        if self.keyword_condition:
            self.null_kw_embedding = nn.Embedding(1, tx_dim)
            self.context_kw_proj = nn.Linear(latent_dim, tx_dim)
        # add projection and null_embedding for reference_condition
        if self.reference_condition:
            self.null_ref_embedding = nn.Embedding(1, tx_dim)
            self.context_ref_proj = nn.Linear(latent_dim, tx_dim)

        self.src_encode_transformer = config.get('src_encode_transformer',False)
        if self.src_encode_transformer:
            print('Using transformer to encode src')
            self.pos_encoder_src_encode_transformer = PositionalEncodingM(self.tx_dim)
            encoder_layers = TransformerEncoderLayer(self.tx_dim,
                                                     8,
                                                     dim_feedforward=2048,
                                                     dropout=0.1,
                                                     )
            self.transformer_src_encoder = TransformerEncoder(encoder_layers, config.transformer_src_encoder_layers)
        else:
            print('Not using transformer to encode src')
        self.audio_encoder_alignment = config.get('audio_encoder_alignment',False)
        if self.audio_encoder_alignment:
            self.joint_embed_shape = config.audio_encoder_alignment_joint_embed_shape
            # add alignment layer after encoder
            self.audio_encoder_alignment_projection = nn.Sequential(
                nn.Linear(self.tx_dim, self.joint_embed_shape),
                nn.ReLU(),
                nn.Linear(self.joint_embed_shape, self.joint_embed_shape),
            )
            self.audio_encoder_alignment_head = MLPLayers([self.joint_embed_shape, self.joint_embed_shape], nn.GELU(), dropout=0.1)
        init_zero_(self.output_proj)

    def forward(self, x, mask, src, time, x_self_cond = None, class_id = None):
        # src: {'waveform': (batch, time)，'kw_latent': (batch, kw_len, latent_dim), 'ref_latent': (batch, ref_len, latent_dim))}
        if self.config.src_encoder.model == 'Extracted':
            if 'audio_features_attn_mask' not in src.keys():
                src['audio_features_attn_mask'] = torch.ones((src['audio_features'].shape[0],src['audio_features'].shape[1]),dtype=torch.bool,device=x.device)
            mem = self.src_encode(src['audio_features'].permute(1,0,2),mask=src['audio_features_attn_mask']) # (b,t,c) -> (t,b,c)
            mem_mask = src['audio_features_attn_mask']
        else:
            mem = self.src_encode(src['waveform'])
            mem_mask = None
        # mem: (time, batch, tx_dim)
        mem = rearrange(mem, 't b d -> b t d')
        if self.audio_encoder_alignment:
            # add alignment layer
            ali_feat = self.audio_encoder_alignment_projection(mem) # (b,t,d) -> (b,t,joint_embed_shape)
            ali_feat = rearrange(ali_feat, 'b t d -> b d t')
            ali_feat = F.adaptive_avg_pool1d(ali_feat, 1).squeeze(-1) # (b,t,joint_embed_shape) -> (b,joint_embed_shape)
            ali_feat = F.normalize(ali_feat, dim=-1)
            ali_feat = self.audio_encoder_alignment_head(ali_feat) # (b,joint_embed_shape) -> (b,joint_embed_shape)
        kw_latent = src.get('kw_latent', None)
        kw_mask = src.get('kw_mask', None)
        latent = self.latent_denoise(x, mask, mem, mem_mask, time, x_self_cond, class_id, kw_latent, kw_mask)
        if self.audio_encoder_alignment:
            return latent, ali_feat # type: ignore
        else:
            return latent

    def src_encode(self, src, mask=None):
        src = self.feature_extractor(src)  # (time, batch, feature)
        src = F.relu_(self.audio_linear(src))
        src = F.dropout(src, p=0.2, training=self.training)
        
        if self.src_encode_transformer:
            src = src * math.sqrt(src.shape[-1])
            src = self.pos_encoder_src_encode_transformer(src)  # time first
            if mask is not None:
                mask_reverse = (mask==False) # torch builtin mask is True for masked position, False for unmasked position
            else:
                mask_reverse = None
            src = self.transformer_src_encoder(src, src_key_padding_mask=mask_reverse)  # time first

        return src  # (time, batch, tx_dim)

    def latent_denoise(self, x, mask, mem, mem_mask, time, x_self_cond = None, class_id = None, kw_latent = None, kw_mask = None):
        """
        x: input, [batch, length, latent_dim]
        mem: encoded audio, [batch, length, nhid]
        mask: bool tensor where False indicates masked positions, [batch, length]
        time: timestep, [batch]
        """

        time_emb = self.time_mlp(time)

        time_emb = rearrange(time_emb, 'b d -> b 1 d')

        pos_emb = self.pos_emb(x)

        tx_input = self.input_proj(x) + pos_emb + self.time_pos_embed_mlp(time_emb)
        if self.cross:
            context, context_mask = [], []
            if mem is None:
                pass
            else:
                # audio conditional context
                if self.req_audio_token_type_emb:
                    mem = mem + self.audio_token_type_embedding(torch.zeros((mem.shape[0],mem.shape[1]),dtype=torch.long,device=x.device))
                if self.req_audio_pos_emb:
                    mem = mem + self.audio_pos_emb(mem)
                context.append(mem)
                if mem_mask is None:
                    mem_mask = torch.ones((mem.shape[0],mem.shape[1]),device=x.device).bool()
                context_mask.append(mem_mask)
            # add keyword conditional context, no pos_emb for keyword, need token_type_emb
            if self.keyword_condition:
                if kw_latent is None:
                    null_context = repeat(self.null_kw_embedding.weight, '1 d -> b 1 d', b=x.shape[0])
                    kw_token_type_emb = self.keyword_condition_token_type_embedding(torch.zeros((null_context.shape[0],null_context.shape[1]), dtype=torch.long, device=x.device))
                    null_context = null_context + kw_token_type_emb
                    context.append(null_context)
                    context_mask.append(torch.tensor([[True] for _ in range(x.shape[0])], dtype=torch.bool, device=x.device))
                else:
                    kw_context = self.context_kw_proj(kw_latent)
                    token_type_emb = self.keyword_condition_token_type_embedding(torch.zeros((kw_context.shape[0],kw_context.shape[1]), dtype=torch.long, device=x.device))
                    kw_context = kw_context + token_type_emb
                    context.append(kw_context)
                    context_mask.append(kw_mask)

            if self.self_condition:
                if x_self_cond is None:
                    null_context = repeat(self.null_embedding.weight, '1 d -> b 1 d', b=x.shape[0])
                    null_context = null_context + self.self_condition_token_type_embedding(torch.zeros((null_context.shape[0],null_context.shape[1]), dtype=torch.long, device=x.device))
                    if self.req_self_condition_pos_emb:
                        null_context = null_context + self.pos_emb(null_context)
                    context.append(null_context)
                    context_mask.append(torch.tensor([[True] for _ in range(x.shape[0])], dtype=torch.bool, device=x.device))
                else:
                    self_cond_context = self.context_proj(x_self_cond)
                    self_cond_context = self_cond_context + self.self_condition_token_type_embedding(torch.zeros((self_cond_context.shape[0],self_cond_context.shape[1]), dtype=torch.long, device=x.device))
                    if self.req_self_condition_pos_emb:
                        self_cond_context = self_cond_context + self.pos_emb(self_cond_context)
                    context.append(self_cond_context)
                    context_mask.append(mask)
            if self.class_conditional:
                assert exists(class_id)
                class_emb = self.class_embedding(class_id)
                class_emb = rearrange(class_emb, 'b d -> b 1 d')
                context.append(class_emb)
                context_mask.append(torch.tensor([[True] for _ in range(x.shape[0])], dtype=bool, device=x.device))
            context = torch.cat(context, dim=1)
            context_mask = torch.cat(context_mask, dim=1)
            # print('shape check (tx_input.shape, mask.shape, context.shape, context_mask.shape, time_emb.shape): ', tx_input.shape, mask.shape, context.shape, context_mask.shape, time_emb.shape)
            x = self.encoder(tx_input, mask=mask, context=context, context_mask=context_mask, time_emb=time_emb)
        else:
            x = self.encoder(tx_input, mask=mask, time_emb=time_emb)

        x = self.norm(x)

        return self.output_proj(x)
