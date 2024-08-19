import os
import sys
sys.path.append(os.getcwd())
from tqdm import tqdm
from transformers import BartTokenizer
from data.DataLoader import AudioCaptionDataset
import json
# load dataset
tokenizer = BartTokenizer.from_pretrained('./pretrained_models/bart-base')
dataset_train = AudioCaptionDataset(tokenizer, max_seq_len=48, dataset='Clotho', split='train', task='captioning', return_dict=False)
# build vocab
vocab = set()
for i in tqdm(range(len(dataset_train))):
    data_item = dataset_train[i]
    cap = data_item['caption']
    tokenized = tokenizer.encode(cap) # list of int
    for token in tokenized:
        vocab.add(token)
banned_tokens = [token for token in range(len(tokenizer.encoder)) if token not in vocab]
# save tokens
with open('./pretrained_models/bart-base-Clotho/vocab_list.json', 'w') as f:
    json.dump(sorted(list(vocab)), f)
# save banned_tokens
with open('./pretrained_models/bart-base-Clotho/vocab_banned_list.json', 'w') as f:
    json.dump(sorted(banned_tokens), f)