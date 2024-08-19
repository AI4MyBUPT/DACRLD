#!/usr/bin/env python3
# coding: utf-8
# @Author  : Xinhao Mei @CVSSP, University of Surrey
# @E-mail  : x.mei@surrey.ac.uk

"""
Adapted from PANNs (Pre-trained Audio Neural Networks).
https://github.com/qiuqiangkong/audioset_tagging_cnn/blob/master/pytorch/models.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torchlibrosa.stft import Spectrogram, LogmelFilterBank
from torchlibrosa.augmentation import SpecAugmentation
import math


def init_layer(layer):
    """Initialize a Linear or Convolutional layer. """
    nn.init.xavier_uniform_(layer.weight)

    if hasattr(layer, 'bias'):
        if layer.bias is not None:
            layer.bias.data.fill_(0.)


def init_bn(bn):
    """Initialize a Batchnorm layer. """
    bn.bias.data.fill_(0.)
    bn.weight.data.fill_(1.)


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):

        super(ConvBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=(3, 3), stride=(1, 1),
                               padding=(1, 1), bias=False)

        self.conv2 = nn.Conv2d(in_channels=out_channels,
                               out_channels=out_channels,
                               kernel_size=(3, 3), stride=(1, 1),
                               padding=(1, 1), bias=False)

        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.init_weight()

    def init_weight(self):
        init_layer(self.conv1)
        init_layer(self.conv2)
        init_bn(self.bn1)
        init_bn(self.bn2)

    def forward(self, input, pool_size=(2, 2), pool_type='avg'):

        x = input
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x)))
        if pool_type == 'max':
            x = F.max_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg':
            x = F.avg_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg+max':
            x1 = F.avg_pool2d(x, kernel_size=pool_size)
            x2 = F.max_pool2d(x, kernel_size=pool_size)
            x = x1 + x2
        else:
            raise Exception('Incorrect argument!')

        return x


class Cnn10(nn.Module):
    def __init__(self, config):

        super(Cnn10, self).__init__()

        self.bn0 = nn.BatchNorm2d(64)

        sr = config.wav.sr
        window_size = config.wav.window_size
        hop_length = config.wav.hop_length
        mel_bins = config.wav.mel_bins
        self.dropout = config.training.dropout

        self.spectrogram_extractor = Spectrogram(n_fft=window_size,
                                                 hop_length=hop_length,
                                                 win_length=window_size,
                                                 window='hann',
                                                 center=True,
                                                 pad_mode='reflect',
                                                 freeze_parameters=True)

        self.logmel_extractor = LogmelFilterBank(sr=sr, n_fft=window_size,
                                                 n_mels=mel_bins,
                                                 fmin=50,
                                                 fmax=14000,
                                                 ref=1.0,
                                                 amin=1e-10,
                                                 top_db=None,
                                                 freeze_parameters=True)

        self.is_spec_augment = config.training.spec_augmentation

        if self.is_spec_augment:
            self.spec_augmenter = SpecAugmentation(time_drop_width=64,
                                                   time_stripes_num=2,
                                                   freq_drop_width=8,
                                                   freq_stripes_num=2)

        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)

        self.init_weight()

    def init_weight(self):
        init_bn(self.bn0)

    def forward(self, input):
        """
        Input: (batch_size, data_length)"""

        x = self.spectrogram_extractor(input)  # (batch_size, 1, time_steps, freq_bins)
        x = self.logmel_extractor(x)  # (batch_size, 1, time_steps, mel_bins)

        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)

        if self.training and self.is_spec_augment:
            x = self.spec_augmenter(x)

        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv_block4(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = torch.mean(x, dim=3)  # batch x channel x time

        (x1, _) = torch.max(x, dim=2)  # max in time
        x2 = torch.mean(x, dim=2)  # average in time
        x = x1 + x2  # batch x channel (512)

        return x


class Cnn14(nn.Module):

    def __init__(self, config):

        super(Cnn14, self).__init__()

        self.bn0 = nn.BatchNorm2d(64)

        sr = config.wav.sr
        window_size = config.wav.window_size
        hop_length = config.wav.hop_length
        mel_bins = config.wav.mel_bins
        self.dropout = config.training.dropout

        self.spectrogram_extractor = Spectrogram(n_fft=window_size,
                                                 hop_length=hop_length,
                                                 win_length=window_size,
                                                 window='hann',
                                                 center=True,
                                                 pad_mode='reflect',
                                                 freeze_parameters=True)

        self.logmel_extractor = LogmelFilterBank(sr=sr, n_fft=window_size,
                                                 n_mels=mel_bins,
                                                 fmin=50,
                                                 fmax=14000,
                                                 ref=1.0,
                                                 amin=1e-10,
                                                 top_db=None,
                                                 freeze_parameters=True)

        self.is_spec_augment = config.training.spec_augmentation

        if self.is_spec_augment:
            self.spec_augmenter = SpecAugmentation(time_drop_width=64,
                                                   time_stripes_num=2,
                                                   freq_drop_width=8,
                                                   freq_stripes_num=2)

        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)
        self.conv_block5 = ConvBlock(in_channels=512, out_channels=1024)
        self.conv_block6 = ConvBlock(in_channels=1024, out_channels=2048)

        self.fc1 = nn.Linear(2048, 512, bias=True)

        self.init_weights()

    def init_weights(self):

        init_bn(self.bn0)
        init_layer(self.fc1)

    def forward(self, input):
        """ input: (batch_size, time_steps, mel_bins)"""

        x = self.spectrogram_extractor(input)  # (batch_size, 1, time_steps, freq_bins)
        x = self.logmel_extractor(x)  # (batch_size, 1, time_steps, mel_bins)

        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)

        if self.training and self.is_spec_augment:
            x = self.spec_augmenter(x)

        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block4(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block5(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block6(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)

        x = torch.mean(x, dim=3)  # batch x channel x time

        (x1, _) = torch.max(x, dim=2)  # max in time
        x2 = torch.mean(x, dim=2)  # average in time
        x = x1 + x2  # batch x channel (2048)
        return x


def _resnet_conv3x3(in_planes, out_planes):
    #3x3 convolution with padding
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1,
                     padding=1, groups=1, bias=False, dilation=1)


def _resnet_conv1x1(in_planes, out_planes):
    #1x1 convolution
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, bias=False)


class _ResnetBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(_ResnetBasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('_ResnetBasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in _ResnetBasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1

        self.stride = stride

        self.conv1 = _resnet_conv3x3(inplanes, planes)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = _resnet_conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

        self.init_weights()

    def init_weights(self):
        init_layer(self.conv1)
        init_bn(self.bn1)
        init_layer(self.conv2)
        init_bn(self.bn2)
        nn.init.constant_(self.bn2.weight, 0)

    def forward(self, x):
        identity = x

        if self.stride == 2:
            out = F.avg_pool2d(x, kernel_size=(2, 2))
        else:
            out = x

        out = self.conv1(out)
        out = self.bn1(out)
        out = self.relu(out)
        out = F.dropout(out, p=0.2, training=self.training)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(identity)

        out += identity
        out = self.relu(out)

        return out


class _ResNet(nn.Module):
    def __init__(self, block, layers, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(_ResNet, self).__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated coqnvolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group

        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            if stride == 1:
                downsample = nn.Sequential(
                    _resnet_conv1x1(self.inplanes, planes * block.expansion),
                    norm_layer(planes * block.expansion),
                )
                init_layer(downsample[0])
                init_bn(downsample[1])
            elif stride == 2:
                downsample = nn.Sequential(
                    nn.AvgPool2d(kernel_size=2),
                    _resnet_conv1x1(self.inplanes, planes * block.expansion),
                    norm_layer(planes * block.expansion),
                )
                init_layer(downsample[1])
                init_bn(downsample[2])

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x


class ResNet38(nn.Module):
    def __init__(self, config):

        super(ResNet38, self).__init__()
        self.bn0 = nn.BatchNorm2d(64)

        sr = config.wav.sr
        window_size = config.wav.window_size
        hop_length = config.wav.hop_length
        mel_bins = config.wav.mel_bins
        self.dropout = config.training.dropout

        self.spectrogram_extractor = Spectrogram(n_fft=window_size,
                                                 hop_length=hop_length,
                                                 win_length=window_size,
                                                 window='hann',
                                                 center=True,
                                                 pad_mode='reflect',
                                                 freeze_parameters=True)

        self.logmel_extractor = LogmelFilterBank(sr=sr, n_fft=window_size,
                                                 n_mels=mel_bins,
                                                 fmin=50,
                                                 fmax=14000,
                                                 ref=1.0,
                                                 amin=1e-10,
                                                 top_db=None,
                                                 freeze_parameters=True)

        self.is_spec_augment = config.training.spec_augmentation

        if self.is_spec_augment:
            self.spec_augmenter = SpecAugmentation(time_drop_width=64,
                                                   time_stripes_num=2,
                                                   freq_drop_width=8,
                                                   freq_stripes_num=2)

        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
        # self.conv_block2 = ConvBlock(in_channels=64, out_channels=64)

        self.resnet = _ResNet(block=_ResnetBasicBlock, layers=[3, 4, 6, 3], zero_init_residual=True)

        self.conv_block_after1 = ConvBlock(in_channels=512, out_channels=2048)

        self.init_weights()

    def init_weights(self):
        init_bn(self.bn0)

    def forward(self, input):
        """
        Input: (batch_size, data_length)"""

        x = self.spectrogram_extractor(input)  # (batch_size, 1, time_steps, freq_bins)
        x = self.logmel_extractor(x)  # (batch_size, 1, time_steps, mel_bins)

        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)

        if self.training and self.is_spec_augment:
            x = self.spec_augmenter(x)

        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=self.dropout, training=self.training, inplace=True)
        x = self.resnet(x)
        x = F.avg_pool2d(x, kernel_size=(2, 2))
        x = F.dropout(x, p=self.dropout, training=self.training, inplace=True)
        x = self.conv_block_after1(x, pool_size=(1, 1), pool_type='avg')
        x = F.dropout(x, p=self.dropout, training=self.training, inplace=True)
        x = torch.mean(x, dim=3)  # batch x channel x time

        (x1, _) = torch.max(x, dim=2)  # max in time
        x2 = torch.mean(x, dim=2)  # average in time
        x = x1 + x2  # batch x channel (512)
        # x = F.relu_(self.fc1(x))
        # x = F.dropout(x, p=self.dropout, training=self.training)

        return x

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


class TFEnc_FeatInput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.tx_dim = config.get('TF_dim',768)
        self.pos_encoder_src_encode_transformer = PositionalEncodingM(self.tx_dim)
        encoder_layers = TransformerEncoderLayer(self.tx_dim,
                                                 8,
                                                 dim_feedforward=2048,
                                                 dropout=0.1,
                                                 )
        self.transformer_src_encoder = TransformerEncoder(encoder_layers, config.transformer_src_encoder_layers)
        audio_feat_extract_dim = config.get('audio_feat_extract_dim', 768)
        self.audio_linear = nn.Linear(audio_feat_extract_dim,self.tx_dim)
        self.cls_token_embedding = nn.Embedding(1, self.tx_dim)

    def forward1(self, x, mask=None):
        """
        x shape: batch, time, channel
        mask shape: batch, time
        """
        batch_size = x.shape[0]
        x = x.permute(1,0,2) # (b,t,c) -> (t,b,c)
        x = self.audio_linear(x)
        x = F.dropout(x, p=0.2, training=self.training)
        x = x * math.sqrt(x.shape[-1])
        x = torch.cat((self.cls_token_embedding.weight.unsqueeze(0).repeat(1,batch_size,1),x),dim=0).to(x.device)
        x = self.pos_encoder_src_encode_transformer(x)  # time first
        if mask is not None:
            mask_reverse = (mask==False) # torch builtin mask is True for masked position, False for unmasked position
            mask_reverse = torch.cat((torch.zeros((mask_reverse.shape[0],1),dtype=torch.bool,device=mask_reverse.device),mask_reverse),dim=1).to(x.device)
            x = self.transformer_src_encoder(x, src_key_padding_mask=mask_reverse)  # time first (t,b,c)
        else:
            x = self.transformer_src_encoder(x,)  # time first (t,b,c)
        # mean max pool + CLS token
        x_cls = x[0,:,:] # batch x channel
        x_contents = x[1:,:,:]
        (x1, _) = torch.max(x_contents, dim=0)  # max in time
        x2 = torch.mean(x_contents, dim=0)  # average in time
        x = (x1 + x2 + x_cls)/3  # batch x channel
        # x = x_cls
        return x
    
    def forward2(self, x, mask=None):
        """
        x shape: batch, time, channel
        mask shape: batch, time
        """
        batch_size = x.shape[0]
        x = x.permute(1,0,2) # (b,t,c) -> (t,b,c)
        x = self.audio_linear(x)
        x = F.dropout(x, p=0.2, training=self.training)
        x = x * math.sqrt(x.shape[-1])
        x = self.pos_encoder_src_encode_transformer(x)  # time first
        if mask is not None:
            mask_reverse = (mask==False) # torch builtin mask is True for masked position, False for unmasked position
            x = self.transformer_src_encoder(x, src_key_padding_mask=mask_reverse)  # time first (t,b,c)
        else:
            x = self.transformer_src_encoder(x,)  # time first (t,b,c)
        # mean max pool + CLS token
        (x1, _) = torch.max(x, dim=0)  # max in time
        x2 = torch.mean(x, dim=0)  # average in time
        x = (x1 + x2)/2  # batch x channel
        # x = x_cls
        return x
    
    def forward(self, x, mask=None):
        """
        x shape: batch, time, channel
        mask shape: batch, time
        """
        batch_size = x.shape[0]
        x = x.permute(1,0,2) # (b,t,c) -> (t,b,c)
        x = self.audio_linear(x)
        x = F.dropout(x, p=0.2, training=self.training)
        x = x * math.sqrt(x.shape[-1])
        x = torch.cat((self.cls_token_embedding.weight.unsqueeze(0).repeat(1,batch_size,1),x),dim=0).to(x.device)
        x = self.pos_encoder_src_encode_transformer(x)  # time first
        if mask is not None:
            mask_reverse = (mask==False) # torch builtin mask is True for masked position, False for unmasked position
            mask_reverse = torch.cat((torch.zeros((mask_reverse.shape[0],1),dtype=torch.bool,device=mask_reverse.device),mask_reverse),dim=1).to(x.device)
            x = self.transformer_src_encoder(x, src_key_padding_mask=mask_reverse)  # time first (t,b,c)
        else:
            x = self.transformer_src_encoder(x,)  # time first (t,b,c)
        # mean max pool + CLS token
        x_cls = x[0,:,:] # batch x channel
        x_contents = x[1:,:,:]
        (x1, _) = torch.max(x_contents, dim=0)  # max in time
        x2 = torch.mean(x_contents, dim=0)  # average in time
        # x = (x1 + x2 + x_cls)/3  # batch x channel
        x = x_cls
        return x
