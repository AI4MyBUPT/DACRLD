"""
use the pretrained model:BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt1.pt
"""
import os
import h5py
import librosa
import numpy as np
from tqdm import tqdm
import torch
import pickle as pkl
from BEATs import BEATs, BEATsConfig

input_dir = 'data/AudioCaps/hdf5s/'
output_dir = 'data/AudioCaps/features/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt1/'
sampling_rate = 16000
audio_duration = 10
feat_dim = 768
device = 'cuda:8'
# load model, strip prediction head
checkpoint = torch.load('pretrained_models/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt1.pt')
cfg = BEATsConfig(checkpoint['cfg'])
BEATs_model = BEATs(cfg)
BEATs_model.load_state_dict(checkpoint['model'])
BEATs_model.eval()
BEATs_model.to(device)
BEATs_model.predictor = None

splits = ['train', 'val', 'test']
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
for split in splits:
    print(f'processing {split}...')
    input_file = os.path.join(input_dir, f'{split}/{split}.h5')
    with h5py.File(input_file, 'r') as hf_raw:
        max_audio_length = hf_raw['waveform'].shape[1]# audio_duration * sampling_rate
        with torch.no_grad():
            for i in tqdm(range(hf_raw['audio_name'].shape[0])):
                audio_length = hf_raw['audio_length'][i]
                audio_name = hf_raw['audio_name'][i]
                waveform_input = hf_raw['waveform'][i]
                waveform_input = torch.from_numpy(waveform_input).unsqueeze(0).to(device)
                audio_feature_lengths = audio_length
                mask = torch.zeros_like(waveform_input).bool()
                # fill unwanted positions with 1
                mask[:,audio_length:] = 1
                features, mask_new = BEATs_model.extract_features(waveform_input, mask)
                output_file = os.path.join(output_dir, f'{split}/{audio_name.decode()}.pkl')
                out_data = [features.squeeze(0).cpu().numpy(),mask_new.logical_not().sum().item()]
                with open(output_file, 'wb') as f:
                    pkl.dump(out_data, f)