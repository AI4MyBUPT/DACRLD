# calculate the mean and std of BART embedding of first batch of train samples
import sys;import os;sys.path.append(os.path.dirname(os.path.abspath(__file__))+'/..')
import torch
from transformers import get_scheduler, AutoTokenizer, PreTrainedTokenizerBase, AutoConfig
from transformers.models.bart.modeling_bart import BartForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutput

from data.DataLoader import build_dataloader
from easydict import EasyDict as edict
import pickle as pkl
config = edict({'model':{'max_seq_len':35,'pretrained_enc_dec_model':'./pretrained_models/bart-base',},'dataset':'AudioCaps','data':{'batch_size':30,'num_workers':1}})
bart_model = BartForConditionalGeneration.from_pretrained(config.model.pretrained_enc_dec_model) # type: ignore
tokenizer = AutoTokenizer.from_pretrained(config.model.pretrained_enc_dec_model) # type: ignore
train_loader = build_dataloader('train',config,tokenizer,return_dict=False)

data = next(iter(train_loader))
input_ids = data['input_ids']
attention_mask = data['attention_mask']
latent = bart_model.get_encoder()(input_ids = input_ids, attention_mask = attention_mask).last_hidden_state # type: ignore
latent_vecs = torch.cat([latent[i][:torch.sum(data['attention_mask'][i])] for i in range(latent.shape[0])], dim=0)
latent_mean = torch.mean(latent_vecs, dim=0)
latent_scale = torch.std(latent_vecs-latent_mean, unbiased=False)
# save the mean and std
with open('pretrained_models/ase_embinput/ac/ac_ase_mean_scale.pkl','wb') as f:
    pkl.dump([latent_mean,latent_scale],f)