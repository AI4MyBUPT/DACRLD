#!/usr/bin/env python3
# coding: utf-8
# adapted from https://github.com/XinhaoMei/DCASE2021_task6_v2.git by Xinhao Mei


import torch
import random
import numpy as np
import h5py
import pickle as pkl
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader


class AudioCaptionDataset(Dataset):

    def __init__(self, tokenizer, max_seq_len, dataset='AudioCaps', split='train', task='captioning', return_dict=False, h5_with_feat_path=None, text_features_pkl_path=None):
        """
        load audio clip's waveform and corresponding caption
        Args:
            dataset: 'AudioCaps', 'Clotho
            split: 'train', 'val', 'test'
        """
        super(AudioCaptionDataset, self).__init__()
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.dataset = dataset
        self.split = split
        self.task = task
        self.return_dict = return_dict
        self.h5_path = f'data/{dataset}/hdf5s/{split}/{split}.h5'
        self.text_features_pkl = None
        if text_features_pkl_path is not None:
            with open(text_features_pkl_path.format(split=split),'rb') as f:
                self.text_features_pkl = pkl.load(f)
        if h5_with_feat_path is not None:
            self.h5_path = h5_with_feat_path.format(split=split)
        if dataset == 'AudioCaps' and split == 'train':
            self.is_train = True
            self.num_captions_per_audio = 1
            with h5py.File(self.h5_path, 'r') as hf:
                self.audio_keys = [audio_name.decode() for audio_name in hf['audio_name'][:]]
                # audio_names: [str]
                self.captions = [caption.decode() for caption in hf['caption'][:]]
        else:
            self.is_train = False
            self.num_captions_per_audio = 5
            with h5py.File(self.h5_path, 'r') as hf:
                self.audio_keys = [audio_name.decode() for audio_name in hf['audio_name'][:]]
                self.captions = [caption for caption in hf['caption'][:]]
                if dataset == 'Clotho':
                    self.audio_lengths = [length for length in hf['audio_length'][:]]
                # [cap_1, cap_2, ..., cap_5]

    def __len__(self):
        if self.task == 'captioning' and self.split != 'train' and self.return_dict:
            return len(self.audio_keys)
        else:
            return len(self.audio_keys) * self.num_captions_per_audio

    def __getitem__(self, index):

        if self.task == 'captioning' and self.split != 'train' and self.return_dict:
            audio_idx = index
        else:
            audio_idx = index // self.num_captions_per_audio
        audio_name = self.audio_keys[audio_idx]
        audio_feature = None
        length_feat = None
        with h5py.File(self.h5_path, 'r') as hf:
            waveform = hf['waveform'][audio_idx]
            if 'audio_features' in hf:
                audio_feature = hf['audio_features'][audio_idx]
                length_feat = hf['audio_feature_lengths'][audio_idx]
                audio_feature = audio_feature[:length_feat]
        text_feature = None
        if self.dataset == 'AudioCaps' and self.is_train:
            caption = self.captions[audio_idx]
            tokenized = self.tokenizer(caption, padding="max_length", truncation=True, max_length=self.max_seq_len, return_tensors='pt')
            input_ids = tokenized['input_ids']
            attention_mask = tokenized['attention_mask']
            if self.text_features_pkl is not None:
                text_feature = torch.from_numpy(self.text_features_pkl[audio_name]).unsqueeze(0)
        else:
            captions = self.captions[audio_idx]
            if self.task == 'captioning' and self.split != 'train' and self.return_dict:
                caption_field = ['caption_{}'.format(i) for i in range(1, self.num_captions_per_audio + 1)]
                caption = {}
                input_ids = {}
                attention_mask = {}
                for i, cap_ind in enumerate(caption_field):
                    caption[cap_ind] = captions[i].decode()
                    tokenized = self.tokenizer(caption[cap_ind], padding="max_length", truncation=True, max_length=self.max_seq_len, return_tensors='pt')
                    input_ids[cap_ind] = tokenized['input_ids']
                    attention_mask[cap_ind] = tokenized['attention_mask']
                if self.text_features_pkl is not None:
                    text_feature = {}
                    for i, cap_ind in enumerate(caption_field):
                        text_feature[cap_ind] = torch.from_numpy(self.text_features_pkl[audio_name][i]).unsqueeze(0)
            else:
                cap_idx = index % self.num_captions_per_audio
                caption = captions[cap_idx].decode()
                tokenized = self.tokenizer(caption, padding="max_length", truncation=True, max_length=self.max_seq_len, return_tensors='pt')
                input_ids = tokenized['input_ids']
                attention_mask = tokenized['attention_mask']
                if self.text_features_pkl is not None:
                    text_feature = torch.from_numpy(self.text_features_pkl[audio_name][cap_idx]).unsqueeze(0)
        if self.dataset == 'Clotho':
            length = self.audio_lengths[audio_idx]
            return {'waveform': waveform, 'caption': caption, 'input_ids': input_ids, 'attention_mask': attention_mask, 'audio_idx': audio_idx,'length': length,
                    'index': index, 'audio_name': audio_name, 'audio_feature': audio_feature, 'length_feat': length_feat,'text_feature': text_feature}
        else:
            return {'waveform': waveform, 'caption': caption, 'input_ids': input_ids, 'attention_mask': attention_mask, 'audio_idx': audio_idx,'length': len(waveform),
                    'index': index, 'audio_name': audio_name,'audio_feature': audio_feature, 'length_feat': length_feat,'text_feature': text_feature}


def collate_fn(batch_data):
    """

    Args:
        batch_data:

    Returns:

    """

    max_audio_length = max([i['length'] for i in batch_data])
    if batch_data[0]['length_feat'] is None:
        max_audio_length_feat = None
    else:
        max_audio_length_feat = max([i['length_feat'] for i in batch_data]) 

    wav_tensor = []
    aud_feat_tensor = []
    audio_features_attn_mask_tensor = []
    audio_features = None
    audio_features_attn_mask = None
    for data in batch_data:
        if max_audio_length > data['waveform'].shape[0]:
            padding = torch.zeros(max_audio_length - data['waveform'].shape[0]).float()
            temp_audio = torch.cat([torch.from_numpy(data['waveform']).float(), padding])
        else:
            temp_audio = torch.from_numpy(data['waveform'][:max_audio_length]).float()
        wav_tensor.append(temp_audio.unsqueeze_(0))
    if batch_data[0]['audio_feature'] is not None:
        for data in batch_data:
            if max_audio_length_feat > data['audio_feature'].shape[0]:
                padding = torch.zeros((max_audio_length_feat - data['audio_feature'].shape[0],data['audio_feature'].shape[1])).float()
                temp_audio_feature_attn_mask = torch.zeros(max_audio_length_feat,dtype=torch.bool) # type: ignore
                temp_audio_feature_attn_mask[:data['audio_feature'].shape[0]] = 1
                temp_audio = torch.cat([torch.from_numpy(data['audio_feature']).float(), padding])
            else:
                temp_audio_feature_attn_mask = torch.ones(max_audio_length_feat,dtype=torch.bool) # type: ignore
                temp_audio = torch.from_numpy(data['audio_feature'][:max_audio_length_feat]).float()
            aud_feat_tensor.append(temp_audio.unsqueeze_(0))
            audio_features_attn_mask_tensor.append(temp_audio_feature_attn_mask.unsqueeze_(0))
        audio_features = torch.cat(aud_feat_tensor)
        audio_features_attn_mask = torch.cat(audio_features_attn_mask_tensor)

    wavs_tensor = torch.cat(wav_tensor)
    captions = [i['caption'] for i in batch_data]
    input_ids = torch.cat([i['input_ids'] for i in batch_data], dim=0)
    attention_mask = torch.cat([i['attention_mask'] for i in batch_data], dim=0)
    audio_ids = torch.Tensor([i['audio_idx'] for i in batch_data])
    indexs = np.array([i['index'] for i in batch_data])
    audio_names = [i['audio_name'] for i in batch_data]
    text_features = None
    if batch_data[0]['text_feature'] is not None:
        text_features = torch.cat([i['text_feature'] for i in batch_data], dim=0)

    return {'waveform': wavs_tensor, 'caption': captions, 'input_ids': input_ids, 'attention_mask': attention_mask, 'audio_idx': audio_ids,'length': len(wavs_tensor),
            'index': indexs, 'audio_name': audio_names, 'audio_features': audio_features, 'audio_features_attn_mask':audio_features_attn_mask, 'text_features': text_features}

def collate_fn_return_dict(batch_data):
    """

    Args:
        batch_data:

    Returns:

    """

    max_audio_length = max([i['length'] for i in batch_data])
    if batch_data[0]['length_feat'] is None:
        max_audio_length_feat = None
    else:
        max_audio_length_feat = max([i['length_feat'] for i in batch_data]) 
 
    wav_tensor = []
    aud_feat_tensor = []
    audio_features_attn_mask_tensor = []
    audio_features = None
    audio_features_attn_mask = None
    for data in batch_data:
        if max_audio_length > data['waveform'].shape[0]:
            padding = torch.zeros(max_audio_length - data['waveform'].shape[0]).float()
            temp_audio = torch.cat([torch.from_numpy(data['waveform']).float(), padding])
        else:
            temp_audio = torch.from_numpy(data['waveform'][:max_audio_length]).float()
        wav_tensor.append(temp_audio.unsqueeze_(0))
    if batch_data[0]['audio_feature'] is not None:
        for data in batch_data:
            if max_audio_length_feat > data['audio_feature'].shape[0]:
                padding = torch.zeros((max_audio_length_feat - data['audio_feature'].shape[0],data['audio_feature'].shape[1])).float()
                temp_audio_feature_attn_mask = torch.zeros(max_audio_length_feat,dtype=torch.bool) # type: ignore
                temp_audio_feature_attn_mask[:data['audio_feature'].shape[0]] = 1
                temp_audio = torch.cat([torch.from_numpy(data['audio_feature']).float(), padding])
            else:
                temp_audio_feature_attn_mask = torch.ones(max_audio_length_feat,dtype=torch.bool) # type: ignore
                temp_audio = torch.from_numpy(data['audio_feature'][:max_audio_length_feat]).float()
            aud_feat_tensor.append(temp_audio.unsqueeze_(0))
            audio_features_attn_mask_tensor.append(temp_audio_feature_attn_mask.unsqueeze_(0))
        audio_features = torch.cat(aud_feat_tensor)
        audio_features_attn_mask = torch.cat(audio_features_attn_mask_tensor)

    wavs_tensor = torch.cat(wav_tensor)
    # captions, input_ids, attention_mask: dict format {'caption_1':'heavy rain with thunder',...}
    captions = [i['caption'] for i in batch_data] # list of dict
    input_ids = [i['input_ids'] for i in batch_data] # list of dict of str:tensor
    attention_mask = [i['attention_mask'] for i in batch_data] # list of dict of str:tensor
    audio_ids = torch.Tensor([i['audio_idx'] for i in batch_data])
    indexs = np.array([i['index'] for i in batch_data])
    audio_names = [i['audio_name'] for i in batch_data]
    text_features = None
    if batch_data[0]['text_feature'] is not None:
        text_features = [i['text_feature'] for i in batch_data]
    return {'waveform': wavs_tensor, 'caption': captions, 'input_ids': input_ids, 'attention_mask': attention_mask, 'audio_idx': audio_ids,'length': len(wavs_tensor),
            'index': indexs, 'audio_name': audio_names, 'audio_features': audio_features, 'audio_features_attn_mask':audio_features_attn_mask, 'text_features': text_features}


def build_dataloader(split, config, tokenizer, return_dict=False):
    dataset = AudioCaptionDataset(tokenizer=tokenizer, max_seq_len=config.model.max_seq_len, dataset=config.dataset, split=split, task='captioning', return_dict=return_dict, h5_with_feat_path=config.data.get('h5_with_feat_path',None),text_features_pkl_path=config.data.get('text_features_pkl_path',None))
    if split == 'train':
        shuffle = True
        drop_last = True
    else:
        shuffle = False
        drop_last = False
    collate_fn_dataloader = collate_fn_return_dict if return_dict else collate_fn
    return DataLoader(dataset=dataset,
                      batch_size=config.data.batch_size,
                      shuffle=shuffle,
                      drop_last=drop_last,
                      num_workers=config.data.num_workers,
                      collate_fn=collate_fn_dataloader)
