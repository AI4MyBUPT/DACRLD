"""
use the pretrained BEATs model: BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt1.pt
"""
import os
import h5py
import librosa
import numpy as np
from tqdm import tqdm
import torch
from BEATs import BEATs, BEATsConfig
import pickle as pkl

input_dir = 'data/Clotho/hdf5s/'
feat_dir = 'data/Clotho/features/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt1/'
output_dir = 'data/Clotho/features/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt1/'
sampling_rate = 16000
audio_duration = 10
feat_dim = 768
device = 'cuda:8'

splits = ['train', 'val', 'test']
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
for split in splits:
    print(f'processing {split}...')
    input_file = os.path.join(input_dir, f'{split}/{split}.h5')
    output_file = os.path.join(output_dir, f'{split}.h5')
    with h5py.File(input_file, 'r') as hf_raw:
        probe_feat_len_file = os.path.join(feat_dir, f'{split}/{hf_raw["audio_name"][0].decode()}.pkl')
        with open(probe_feat_len_file, 'rb') as f:
            probe_feat_len_data = pkl.load(f)
        audio_feature_lengths_max = probe_feat_len_data[0].shape[0]
        with h5py.File(output_file, 'w') as hf_new:
            hf_new.create_dataset('waveform', shape=hf_raw['waveform'].shape, data=hf_raw['waveform'], dtype=np.float32)
            hf_new.create_dataset('audio_length', shape=hf_raw['audio_length'].shape, data=hf_raw['audio_length'], dtype=np.uint32)
            hf_new.create_dataset('caption', shape=hf_raw['caption'].shape, data=hf_raw['caption'],dtype=h5py.special_dtype(vlen=str))
            hf_new.create_dataset('audio_name', shape=hf_raw['audio_name'].shape, data=hf_raw['audio_name'], dtype=h5py.special_dtype(vlen=str))
            hf_new.create_dataset('audio_features', shape=(hf_raw['audio_name'].shape[0],audio_feature_lengths_max,feat_dim),dtype=np.float32)
            hf_new.create_dataset('audio_feature_lengths', shape=(hf_raw['audio_name'].shape[0],),dtype=np.uint32)            
            for i in tqdm(range(hf_raw['audio_name'].shape[0])):
                audio_name = hf_raw['audio_name'][i]
                feat_file = os.path.join(feat_dir, f'{split}/{audio_name.decode()}.pkl')
                with open(feat_file, 'rb') as f:
                    out_data = pkl.load(f)
                hf_new['audio_features'][i] = out_data[0]
                hf_new['audio_feature_lengths'][i] = out_data[1]