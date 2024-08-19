import os, sys, glob, json
sys.path.append(os.getcwd())
import numpy as np
import argparse
import pickle
import torch
import logging

from torchmetrics.text.rouge import ROUGEScore
rougeScore = ROUGEScore()
# from bert_score import score

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import single_meteor_score
import nltk
from rouge import Rouge


from eval_metrics import evaluate_metrics
# eval diversity
from eval_diversity_metrics import evaluate_metrics_diversity

def remove_dup(text_pred):
    text_pred_no_dup = []
    for aud_id in range(len(text_pred)):
        text_pred_no_dup.append([])
        for s in text_pred[aud_id]:
            s = s.split()
            s_no_dup = [s[0]]
            for i in range(1, len(s)):
                if s[i] != s[i - 1]:
                    s_no_dup.append(s[i])
            text_pred_no_dup[aud_id].append(' '.join(s_no_dup))
    return text_pred_no_dup

def get_bleu(recover, reference):
    return sentence_bleu(references=[reference.split()], hypothesis=recover.split(), smoothing_function=SmoothingFunction().method4,)

def get_mbr(sentences,k):
    if k > 0:
        sentences = sentences[:k]
    selfBleu = [[] for i in range(len(sentences))]
    for i, s1 in enumerate(sentences):
        for j, s2 in enumerate(sentences):
            score = get_bleu(s1, s2)
            selfBleu[i].append(score)
    for i, s1 in enumerate(sentences):
        selfBleu[i][i] = 0
    idx = np.argmax(np.sum(selfBleu, -1))
    return sentences[idx]

def coco_eval(pred, ref):
    pred_dict_list = []
    for aud_id in range(len(pred)):
        pred_dict_list.append({'file_name':ref[aud_id]['file_name'],'caption_predicted':pred[aud_id]})
    metrics = evaluate_metrics(prediction_file=pred_dict_list, reference_file=ref)
    for metric, values in metrics.items():
        logging.info(f'{metric:<7s}: {values["score"]:7.4f}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_file', type=str, default=
                        'output/20240115-091658.data_Clotho_latentTF_Extracted_scratch_v0.2_lr_0.0001_sche_cosine_warm_2000_bs_20/eval_20240220-215207_beats_1e-5_norm0924_langevin_coef_0.00001_step_size 0.1_step_count_3/text_ref_fname_test_ep_67.pkl')
    parser.add_argument('--diversity_k', type=int, default=5)
    args = parser.parse_args()
    out_dir = os.path.join(os.path.split(args.in_file)[0],'deduped')
    out_file = os.path.join(out_dir, 'deduped_'+os.path.split(args.in_file)[1])
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    logging.basicConfig(filename=os.path.join(out_dir,'output_dedup.txt'),level=logging.INFO,force=True)
    with open(args.in_file, 'rb') as f:
        data = pickle.load(f)
    text_pred_allgen_method, ref_dict, fname_list = data[0], data[1], data[2]
    text_pred_no_dup_allgen_method = {}
    for gen_method in list(text_pred_allgen_method.keys()):
        text_pred = text_pred_allgen_method[gen_method]
        text_pred_no_dup = remove_dup(text_pred)
        text_pred_no_dup_allgen_method[gen_method] = text_pred_no_dup
    with open(out_file, 'wb') as f:
        pickle.dump([text_pred_no_dup_allgen_method, ref_dict, fname_list], f)
    for gen_method in list(text_pred_no_dup_allgen_method.keys()):
        text_pred_no_dup = text_pred_no_dup_allgen_method[gen_method]
        text_pred_no_dup_used = [item[:args.diversity_k] for item in text_pred_no_dup]
        metrics = evaluate_metrics_diversity(text_pred_no_dup_used)
        logging.info('for generation method {}:'.format(gen_method))
        for metric, values in metrics.items():
            logging.info(f'  {metric:<7s}: {values["score"]:7.4f}')
    logging.info(f'accuracy evaluation:')
    for gen_method in list(text_pred_no_dup_allgen_method.keys()):
        text_pred_no_dup = text_pred_no_dup_allgen_method[gen_method]
        text_pred_eval = []
        try:
            for aud_id in range(len(text_pred_no_dup)):
                text_pred_eval.append(get_mbr(text_pred_no_dup[aud_id],k=50))
        except Exception as e:
            print(e)
            exit()
        logging.info('for generation method {}:'.format(gen_method))
        coco_eval(text_pred_eval, ref_dict)
