import os,sys
sys.path.append(os.getcwd())
import pickle as pkl
import torch
from models.ASE_model import ASE
from tools.config_loader import get_config
from data_handling.DataLoader import get_dataloader_inf
from sentence_transformers import util
from tqdm import tqdm
import numpy as np
DBG=False
model_output_dir = 'outputs/Clotho_Cnn14_b28_1e-4_data_Clotho_freeze_False_lr_0.0001_margin_0.2_seed_20/models'
savepath = 'data/Clotho/xmodaler/clotho_ret_sents_Cnn14_b28_1e-4_margin_0.2_seed_20.pkl'
device = 'cuda:0'
config_path = 'settings_inference'
config = get_config(config_path)
model = ASE(config)
model = model.to(device)
best_checkpoint = torch.load(str(model_output_dir) + '/best_model.pth',map_location=device)
model.load_state_dict(best_checkpoint['model'])
model.eval()
train_loader = get_dataloader_inf('train', config)
val_loader = get_dataloader_inf('val', config)
test_loader = get_dataloader_inf('test', config)

# inference function
def inference(data_loader, model):
    with torch.no_grad():
        # numpy array to keep all embeddings in the dataset
        audio_embs, cap_embs = None, None
        audio_names_list = []
        for i, batch_data in tqdm(enumerate(data_loader), total=len(data_loader)):
            audios, captions, audio_ids, indexs, audio_names = batch_data
            # move data to GPU
            audios = audios.to(device)

            audio_embeds, caption_embeds = model(audios, captions)

            if audio_embs is None:
                audio_embs = np.zeros((len(data_loader.dataset), audio_embeds.size(1)))
                cap_embs = np.zeros((len(data_loader.dataset), caption_embeds.size(1)))

            audio_embs[indexs] = audio_embeds.cpu().numpy()
            cap_embs[indexs] = caption_embeds.cpu().numpy()
            audio_names_list.extend(audio_names)
            if DBG:
                print(audio_embs.shape,cap_embs.shape,audio_names_list)
                audio_names_list = [str(i) for i in range(len(data_loader.dataset))]
                break
    return audio_embs, cap_embs, audio_names_list

class Tokenizer:
    def __init__(self,vocab_path='../open_source_dataset/Clotho/pickles/words_list.p',vocab_txt_path='../open_source_dataset/Clotho/xmodaler/vocabulary.txt'):
        with open(vocab_path,'rb') as f:
            vocab = pkl.load(f)
        self.itow = {i+1:w for i,w in enumerate(vocab)} # a 1-indexed vocab translation table
        self.wtoi = {w:i+1 for i,w in enumerate(vocab)} # inverse table
        # save txt version
        vocab_str = '\n'.join(vocab)
        with open(vocab_txt_path,'w') as f:
            f.write(vocab_str)

    def __call__(self,cap_list,max_len=30):
        input_Li = np.zeros((len(cap_list),max_len),dtype=np.uint32)
        input_mask = np.zeros((len(cap_list),max_len),dtype=np.uint32)
        input_mask[:,0] = 1 # <SOS> always active
        # tokenize
        for idx, cap in enumerate(cap_list):
            cap = cap.strip().split()
            for k, w in enumerate(cap):
                if k < max_len:
                    input_Li[idx,k+1] = self.wtoi[w] # one shift for <BOS>
                    input_mask[idx,k+1] = 1
        return input_Li, input_mask
tokenizer = Tokenizer()

# get embedding for all train, val, test audios
# get embedding for all train captions
audio_embs_tr, cap_embs_tr, audio_names_tr = inference(train_loader,model)
audio_embs_val, cap_embs_val, audio_names_val = inference(val_loader,model)
audio_embs_test, cap_embs_test, audio_names_test = inference(test_loader,model)
# make deduplicated, concatenated audio embedding matrix and name matrix
audio_names = [audio_names_tr[i]for i in range(0, len(audio_names_tr), 5)] + \
              [audio_names_val[i]for i in range(0, len(audio_names_val), 5)] + \
              [audio_names_test[i]for i in range(0, len(audio_names_test), 5)]
cat_audios = np.concatenate([audio_embs_tr,audio_embs_val,audio_embs_test],axis=0)
audios = np.array([cat_audios[i]for i in range(0, cat_audios.shape[0], 5)])
tr_audio_len = int(audio_embs_tr.shape[0]/5)
# cal cos_sim for train+val+test_audios X train_captions
sim_matrix = util.cos_sim(torch.Tensor(audios),torch.Tensor(cap_embs_tr))
# sim_matrix is tensor
# special treatmet for train audios:
# 1.(TODO) see whether clearing self-nomination needed
# 2. sim_matrix[:train_len,:]
for train_audio_idx in range(tr_audio_len):
    sim_matrix[train_audio_idx,5*train_audio_idx:5*(train_audio_idx+1)] = -999
# get rank list for each audio
sim_matrix = sim_matrix.numpy()
sim_rank = np.argsort(sim_matrix,axis=1)[:,::-1]
# make datadict:
# file format: dict, key: str audio_name value: r_tokens_ids: array (20,30) dtype=uint32 r_tokens_mask array (20,30) dtype=uint32 r_clip_scores array (20,) dtype=float32
# save to SCD formats
datadict = {}
for audio_all_idx in range(sim_matrix.shape[0]):
    sim_scores = np.zeros((20,),dtype=np.float32)
    captions = []
    for cap_rank_idx in range(20):
        train_cap_idx = sim_rank[audio_all_idx,cap_rank_idx]
        sim_scores[cap_rank_idx] = sim_matrix[audio_all_idx,train_cap_idx]
        _, caption, _, _, _, audio_name = train_loader.dataset[train_cap_idx]
        captions.append(caption)
    r_tokens_ids,r_tokens_mask = tokenizer(captions)
    datadict[audio_names[audio_all_idx]] = {
        'r_tokens_ids': r_tokens_ids,
        'r_tokens_mask': r_tokens_mask,
        'r_clip_scores': sim_scores
    }
with open(savepath,'wb') as f:
    pkl.dump(datadict,f)