"""
adapted from https://github.com/sabirdvd/DivStats_Caption/blob/main/computeDivStats.py
"""

import json
import numpy as np
import argparse
from typing import List, Dict
import os
import logging
import pickle
if __name__ == '__main__':
    import sys
    sys.path.append(os.getcwd())

from coco_caption.pycocoevalcap.meteor.meteor import Meteor
from coco_caption.pycocoevalcap.bleu.bleu import Bleu
from coco_caption.pycocoevalcap.rouge.rouge import Rouge
from coco_caption.pycocoevalcap.cider.cider import Cider
from coco_caption.pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer


def find_ngrams(input_list, n):
    return zip(*[input_list[i:] for i in range(n)])


def compute_global_div_n(caps: Dict[str,List],n=1):
  aggr_div = []
  all_ngrams = set()
  lenT = 0.
  for k in caps:
      for c in caps[k]:
         tkns = c.split()  # list
         lenT += len(tkns)
         ng = find_ngrams(tkns, n)
         all_ngrams.update(list(ng))
  if n == 1:
    aggr_div.append(float(len(all_ngrams)))
  else:
    aggr_div.append(float(len(all_ngrams)) / (1e-6 + float(lenT)))
  return aggr_div[0], np.repeat(np.array(aggr_div),len(caps))


def compute_div_n(caps,n=1):
  aggr_div = []
  for k in caps:
      all_ngrams = set()
      lenT = 0.
      for c in caps[k]:
         tkns = c.split()
         lenT += len(tkns)
         ng = find_ngrams(tkns, n)
         all_ngrams.update(list(ng))
      aggr_div.append(float(len(all_ngrams))/ (1e-6 + float(lenT)))
  return np.array(aggr_div).mean(), np.array(aggr_div)


def evaluate_metrics_diversity(pred_text_list:List[List],candidate_size=5):
    """
    vocab_size: number of unique words for generated captions in the test set
    :param pred_text_list: [[audio_1_pred_1,audio_1_pred_2,..,audio_1_pred_x],[audio_2_pred_1,..]]
    :return: {'div_1':float, 'div_2':float,'mbleu_4':float, 'vocab_size':int}
    """
    # cut the prediction of each audio to candidate_size
    for data_id in range(len(pred_text_list)):
        pred_text_list[data_id] = pred_text_list[data_id][:candidate_size]
    tokenizer = PTBTokenizer()
    capsById = {}
    for i, preds in enumerate(pred_text_list):
        capsById[i] = [{'image_id':i, 'caption':p, 'id': i*len(preds)+j} for j,p in enumerate(preds)]
    capsById = tokenizer.tokenize(capsById)
    div_1, adiv_1 = compute_div_n(capsById,1)
    div_2, adiv_2 = compute_div_n(capsById,2)
    globdiv_1, _= compute_global_div_n(capsById,1)
    globdiv_2, _= compute_global_div_n(capsById,2)
    # calculate mbleu_4
    # Run 1 vs rest bleu metrics
    all_scrs = []
    n_caps_perimg = len(capsById[list(capsById.keys())[0]])
    scrperimg = np.zeros((n_caps_perimg, len(capsById)))
    for i in range(n_caps_perimg):
        tempRefsById = {}
        candsById = {}
        for k in capsById:
            tempRefsById[k] = capsById[k][:i] + capsById[k][i+1:]
            candsById[k] = [capsById[k][i]]
        scorer = Bleu(4)
        scr, scrs = scorer.compute_score(tempRefsById, candsById)
        all_scrs.append(scr)
        scrperimg[i, :] = scrs[3] # bleu_4
    all_scrs = np.array(all_scrs)
    mbleu = all_scrs.mean(axis=0)
    # calculate vocab_size (for entire test set)
    vocab_size = cal_vocab_size(capsById)
    return {'div_1':{'score': div_1}, 'div_2':{'score': div_2}, 'mbleu_1':{'score':mbleu[0]}, 'mbleu_2':{'score':mbleu[1]}, 'mbleu_3':{'score':mbleu[2]}, 'mbleu_4':{'score':mbleu[3],'scores':scrperimg}, 'vocab_size':{'score':vocab_size}, 'globdiv_1':{'score':globdiv_1}, 'globdiv_2':{'score':globdiv_2}}


def cal_vocab_size(capsById):
    vocab = set()
    for k in capsById: # each audio
        for c in capsById[k]: # each caption
            vocab.update(c.split())
    return len(vocab)

def main_old(params):
 tokenizer = PTBTokenizer()
 for resF in params['resFileList']:
    caps = json.load(open(resF,'r'))
    capsById = {}
    idTocaps = {}
    n_cands = params['keepN'] - 1 if params['keepN'] !=None else None
    n=0
    for i,img in enumerate(caps['imgblobs']):
        imgid = int(img['img_path'].split('_')[-1].split('.')[0])
        capsById[imgid] = [{'image_id':imgid, 'caption':img['candidate']['text'], 'id': n}]
        idTocaps[imgid] = i
        n+=1
        capsById[imgid].extend([{'image_id':imgid, 'caption':cd['text'], 'id': n+j} for j,cd in enumerate(img['candidatelist'][:n_cands])])
        #if len(capsById[imgid]) < (n_cands+1):
        #   capsById[imgid].extend([capsById[imgid][-1] for _ in xrange(n_cands+1 - len(capsById[imgid]))])
        #n+=len(capsById[imgid]) -1

    n_caps_perimg = len(capsById[capsById.keys()[0]])
    # print(n_caps_perimg)
    capsById = tokenizer.tokenize(capsById)

    div_1, adiv_1 = compute_div_n(capsById,1)
    div_2, adiv_2 = compute_div_n(capsById,2)

    globdiv_1, _= compute_global_div_n(capsById,1)

    # print('Diversity Statistics are as follows: \n Div1: %.2f, Div2: %.2f, gDiv1: %d\n'%(div_1,div_2, globdiv_1))

    if params['compute_mbleu']:
        scorer = Bleu(4)

        # Run 1 vs rest bleu metrics
        all_scrs = []
        scrperimg = np.zeros((n_caps_perimg, len(capsById)))
        for i in range(n_caps_perimg):
            tempRefsById = {}
            candsById = {}
            for k in capsById:
                tempRefsById[k] = capsById[k][:i] + capsById[k][i+1:]
                candsById[k] = [capsById[k][i]]

            score, scores = scorer.compute_score(tempRefsById, candsById)
            all_scrs.append(score)
            scrperimg[i,:] = scores[1]

        all_scrs = np.array(all_scrs)
        if params['writeback']:
            for i,imgid in enumerate(capsById.keys()):
                caps['imgblobs'][idTocaps[imgid]]['mBleu_2'] = scrperimg[:,i].mean()
                caps['imgblobs'][idTocaps[imgid]]['candidate']['mBleu_2'] = scrperimg[0,i]
                for j,st in enumerate(caps['imgblobs'][idTocaps[imgid]]['candidatelist'][:n_cands]):
                    caps['imgblobs'][idTocaps[imgid]]['candidatelist'][j]['mBleu_2'] = scrperimg[1+j,i]
            json.dump(caps,open(resF,'w'))


        # print('Mean mutual Bleu scores on this set is:\nmBLeu_1, mBLeu_2, mBLeu_3, mBLeu_4\n')
        # print(all_scrs.mean(axis=0))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_file', type=str, default='output/20230526-005414.data_Clotho_model_DiffusionLatentTransformer_0.1_lr_0.0001_sche_cosine_warm_1000_bs_46/eval_20230921-021539/text_ref_fname_test_ep_74.pkl')
    parser.add_argument('--out', type=str, default=f'output/20230526-005414.data_Clotho_model_DiffusionLatentTransformer_0.1_lr_0.0001_sche_cosine_warm_1000_bs_46/eval_20230921-021539')
    parser.add_argument('--k', type=int, default=5)
    args = parser.parse_args()
    if not os.path.exists(args.out):
        os.mkdir(args.out)
    # setup logger
    logging.basicConfig(filename=os.path.join(args.out,'output_div.txt'),level=logging.INFO)
    with open(args.in_file, 'rb') as f:
        data = pickle.load(f)
    text_pred_allgen_method, ref_dict, fname_list = data[0], data[1], data[2]
    logging.info(f'candidate size is {args.k}')
    for gen_method in list(text_pred_allgen_method.keys()):
        text_pred = text_pred_allgen_method[gen_method]
        text_pred_used = [item[:args.k] for item in text_pred]
        metrics = evaluate_metrics_diversity(text_pred_used)
        logging.info('for generation method {}:'.format(gen_method))
        for metric, values in metrics.items():
            logging.info(f'  {metric:<7s}: {values["score"]:7.4f}')
