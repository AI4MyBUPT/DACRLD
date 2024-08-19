import os
import random
import numpy as np
import torch
from transformers import LogitsProcessor,LogitsProcessorList

class WordListConfiner(LogitsProcessor):
    def __init__(self,word_blacklist,confine_to_word_list=True):
        self.word_blacklist = word_blacklist
        self.confine_to_word_list = confine_to_word_list

    def __call__(self,input_ids, scores):
        if self.confine_to_word_list:
            scores[:,self.word_blacklist] = scores[:,self.word_blacklist]-1e10
        else:
            pass
        return scores

def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    # torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
