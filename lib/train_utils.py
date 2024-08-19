from . import set_logger
from . import get_adamw_optimizer
from . import build_lr_scheduler
from . import compute_metrics
from . import set_seed
from .eval_metrics import evaluate_metrics as evaluate_metrics_coco
from .eval_diversity_metrics import evaluate_metrics_diversity
from .diffusion import GaussianDiffusion
from .file_io import load_pickle_file
from .langevin import LANGEVIN_FN_DICT

from .utils import WordListConfiner
from transformers import LogitsProcessorList
import torch

from src import build_model
from src.loss import NTXent
from data import build_dataloader

from ema_pytorch import EMA
from accelerate import Accelerator
import wandb
from tqdm import tqdm
import pickle as pkl
import json
import os
import time
import math
from typing import List, Dict, Union
from functools import partial
from collections import namedtuple, Counter

from transformers import get_scheduler, AutoTokenizer, PreTrainedTokenizerBase, AutoConfig
from transformers.models.bart.modeling_bart import BartForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutput
import torch
import numpy as np

from transformers.trainer_pt_utils import LabelSmoother
torch.multiprocessing.set_sharing_strategy('file_system')
# partially adapted from https://github.com/lucidrains/denoising-diffusion-pytorch and https://github.com/justinlovelace/latent-diffusion-for-language

def convert_metrics(metrics_dict):
    """
    convert metrics to wandb logging format
    """
    metrics = {}
    all_good = True
    for k, v in metrics_dict.items(): # level: metric name, decoding method name
        if isinstance(v, float) or isinstance(v, int):
            metrics[k] = v
        elif isinstance(v, np.ndarray):
            all_good = False
            if v.ndim == 0:
                metrics[k] = v.item()
            elif v.ndim == 1:
                metrics[k] = v.tolist()
            else:
                print(f'unknown metric type: {v}')
        elif isinstance(v, dict): # score and scores level
            all_good = False
            if 'score' in v.keys():
                metrics[k] = v['score']
            else:
                for k1, v1 in v.items():
                    metrics[f'{k}/{k1}'] = v1
        else:
            metrics[k] = v
            print(f'unknown metric type: {v}')
    return convert_metrics(metrics) if not all_good else metrics

class Trainer(object):
    def __init__(self, cfg, logger=None, outdir=None):
        super().__init__()
        self.device_first = torch.device(f'cuda:{cfg.system.device[0]}')
        torch.cuda.empty_cache()
        set_seed(cfg.system.seed)
        if logger is None or outdir is None:
            print('set logger')
            self.logger, self.outdir = set_logger(cfg)
        else:
            self.logger = logger
            self.outdir = outdir
        self.outdir_inference = os.path.join(self.outdir, 'inference')
        self.cfg = cfg
        self.bartcfg = AutoConfig.from_pretrained(cfg.model.pretrained_enc_dec_model)
        self.cfg.model.latent_dim = self.bartcfg.d_model
        self.model = build_model(cfg.model).to(self.device_first)
        if cfg.diffusion.langevin_sampling.langevin_fn_type:
            self.langevin_need_restore_norm = cfg.diffusion.langevin_sampling.get('langevin_need_restore_norm',False)
            if self.langevin_need_restore_norm:
                if 'ase_mean_scale' in self.cfg.diffusion.langevin_sampling.keys():
                    with open(self.cfg.diffusion.langevin_sampling.ase_mean_scale,'rb') as f:
                        mean_scale = pkl.load(f)
                    self.langevin_need_restore_norm = (torch.tensor(mean_scale[0],device=self.device_first),torch.tensor(mean_scale[1],device=self.device_first))
                else:
                    self.langevin_need_restore_norm = 'bart_space'
        else:
            self.langevin_need_restore_norm = False
        self.diffusion = GaussianDiffusion(self.model,
                                           max_seq_len=cfg.diffusion.max_seq_len,
                                           timesteps=cfg.diffusion.timesteps,
                                           sampling_timesteps=cfg.diffusion.sampling_timesteps,
                                           loss_type=cfg.diffusion.loss_type,
                                           beta_schedule=cfg.diffusion.beta_schedule,
                                           p2_loss_weight_gamma=cfg.diffusion.p2_loss_weight_gamma,
                                           objective=cfg.diffusion.objective,
                                           ddim_sampling_eta=cfg.diffusion.ddim_sampling_eta,
                                           langevin_need_restore_norm=self.langevin_need_restore_norm).to(self.device_first)
        self.diffusion: torch.nn.Module = torch.nn.DataParallel(self.diffusion, device_ids=cfg.system.device).to(self.device_first) # type: ignore
        self.ema = EMA(self.diffusion.module, beta = self.cfg.diffusion.ema_decay, update_every = self.cfg.diffusion.ema_update_every, power=3/4)
        self.ema.to(self.device_first)
        self.ema_model_dp = torch.nn.DataParallel(self.ema.ema_model, device_ids=cfg.system.device).to(self.device_first) # type: ignore
        self.optimizer = get_adamw_optimizer(self.diffusion.parameters(), lr=cfg.optimizer.lr, betas=cfg.optimizer.betas, weight_decay=cfg.optimizer.weight_decay)
        
        self.bart_model = BartForConditionalGeneration.from_pretrained(self.cfg.model.pretrained_enc_dec_model).to(self.device_first) # type: ignore
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model.pretrained_enc_dec_model)
        
        self.train_dataloader = build_dataloader(split='train', config=cfg, tokenizer=self.tokenizer)
        
        num_training_steps = math.ceil(len(self.train_dataloader) / cfg.train.gradient_accumulate_every) * cfg.train.max_epochs
        self.lr_scheduler = get_scheduler(cfg.lr_scheduler.type, optimizer=self.optimizer, num_warmup_steps=cfg.lr_scheduler.num_warmup_steps,num_training_steps=num_training_steps)

        
        self.val_dataloader = build_dataloader(split='val', config=cfg, tokenizer=self.tokenizer, return_dict=False)
        self.val_dataloader_return_dict = build_dataloader(split='val', config=cfg, tokenizer=self.tokenizer, return_dict=True)
        self.test_dataloader = build_dataloader(split='test', config=cfg, tokenizer=self.tokenizer, return_dict=False)
        self.test_dataloader_return_dict = build_dataloader(split='test', config=cfg, tokenizer=self.tokenizer, return_dict=True)
        if not os.path.exists(os.path.split(cfg.data.length_distri_file)[0]):
            os.makedirs(os.path.split(cfg.data.length_distri_file)[0])
        if os.path.exists(cfg.data.length_distri_file):
            self.length_categorical = torch.load(cfg.data.length_distri_file)
        else:
            print('calculate length distribution')
            train_val_lengths = [min(self.train_dataloader.dataset[idx]['attention_mask'].sum().item(), cfg.diffusion.max_seq_len) for idx in range(len(self.train_dataloader.dataset))]# type: ignore
            train_val_lengths.extend([min(self.val_dataloader.dataset[idx]['attention_mask'].sum().item(), cfg.diffusion.max_seq_len) for idx in range(len(self.val_dataloader.dataset))])# type: ignore
            length_counts = Counter(train_val_lengths)
            probs = torch.tensor([length_counts[idx]/len(train_val_lengths) for idx in range(cfg.diffusion.max_seq_len+1)])
            self.length_categorical = torch.distributions.Categorical(probs=probs)
            torch.save(self.length_categorical, cfg.data.length_distri_file)
        self.epoch = 0
        self.max_epochs = self.cfg.train.max_epochs # epoch-based
        self.gradient_accumulate_every = self.cfg.train.gradient_accumulate_every

        self.is_keyword = self.cfg.data.is_keyword
        if self.is_keyword is True:
            num_keywords = self.cfg.data.num_keywords
            train_keywords_dict = load_pickle_file('data/Clotho/pickles/456/train_keywords_dict_pred.p')
            train_size = len(self.train_dataloader.dataset)# type: ignore
            self.keywords_list = []
            for i in range(0, train_size, 5):
                file_name = self.train_dataloader.dataset[i]['audio_name']
                keywords = train_keywords_dict[file_name]
                keywords_tok = self.tokenizer(' '+' '.join(keywords))
                self.keywords_list.append(keywords_tok)

            # val_keywords, test_keywords = compute_keywords(config, train_loader, val_loader, test_loader, keywords_list)
            val_keywords_dict = load_pickle_file('data/Clotho/pickles/456/val_keywords_dict_pred.p'.format(num_keywords))
            val_size = len(self.val_dataloader.dataset)# type: ignore
            self.val_keywords = []
            for i in range(val_size):
                file_name = self.val_dataloader.dataset[i]['audio_name']
                keywords = val_keywords_dict[file_name]
                keywords_tok = self.tokenizer(' '+' '.join(keywords))
                self.val_keywords.append(keywords_tok)

            test_keywords_dict = load_pickle_file('data/Clotho/pickles/456/test_keywords_dict_pred.p'.format(num_keywords))
            test_size = len(self.test_dataloader.dataset)# type: ignore
            self.test_keywords = []
            for i in range(test_size):
                file_name = self.test_dataloader.dataset[i]['audio_name']
                keywords = test_keywords_dict[file_name]
                keywords_tok = self.tokenizer(' '+' '.join(keywords))
                self.test_keywords.append(keywords_tok)
        self.best_tracker = {}
        self.best_tracker['val'] = {k:{'epoch':0,'perf':None} for k in self.cfg.train.monitor_metrics}
        self.best_tracker['test'] = {k:{'epoch':0,'perf':None} for k in self.cfg.train.monitor_metrics}
        self.no_improvement_count = 0
        # langevin sampling
        self.langevin_fn = None
        if self.cfg.diffusion.langevin_sampling.langevin_fn_type:
            self.langevin_fn = LANGEVIN_FN_DICT[self.cfg.diffusion.langevin_sampling.langevin_fn_type]
            from src.ASE_model_embinput import ASE
            self.langevin_control_model = ASE(self.cfg.diffusion.langevin_sampling.ase_model).to(device=self.device_first)
            self.langevin_control_model.load_state_dict(torch.load(self.cfg.diffusion.langevin_sampling.langevin_control_model_path,map_location=self.device_first)['model'])
            self.langevin_control_model.eval()
            # fill the known parameters with partial
            self.langevin_fn = partial(self.langevin_fn,
                                       retrieval_model=self.langevin_control_model,
                                       config={'audio_src_type':self.cfg.diffusion.langevin_sampling.get('audio_src_type','waveform'),},
                                       step_size=self.cfg.diffusion.langevin_sampling.langevin_step_size,
                                       i_max=self.cfg.diffusion.langevin_sampling.langevin_step_count,
                                       mag_hyper=self.cfg.diffusion.langevin_sampling.langevin_coeff)
        self.audio_encoder_alignment = cfg.model.get('audio_encoder_alignment',False)
        if self.audio_encoder_alignment:
            self.alignment_loss = NTXent()

        if self.cfg.diffusion.get('ce_loss',None):
            if self.cfg.diffusion.ce_loss.get('label_smoothing_factor',None):
                self.label_smoother = LabelSmoother(epsilon=self.cfg.diffusion.ce_loss.label_smoothing_factor)
    
    
    def train(self):
        # suppress errors
        assert self.tokenizer.pad_token_id is not None
        assert type(self.ema.ema_model) == GaussianDiffusion
        assert type(self.diffusion.module) == GaussianDiffusion
        assert self.train_dataloader is not None, 'train_dataloader is None'
        if os.path.exists(os.path.join(self.outdir, 'result_metric_all.pkl')):
            with open(os.path.join(self.outdir, 'result_metric_all.pkl'), 'rb') as f:
                result_metric_tracking_all = pkl.load(f)
            tracking_epoch = max(result_metric_tracking_all.keys())+1
            self.logger.info(f'Load result_metric_all.pkl from {self.outdir}, tracking_epoch {tracking_epoch}, epoch {self.epoch}')
        else:
            result_metric_tracking_all = {}
        b_id = 0
        try:
            while self.epoch < self.max_epochs:
                start_time = time.time()
                self.logger.info(f'Epoch [{self.epoch}/{self.max_epochs}] starts.')
                self.diffusion.train() # at the end of each epoch, the model is set to eval. Revert to train here.  
                self.ema.ema_model.train()
                train_loss_sum = 0 # for logging
                t = tqdm(self.train_dataloader, ncols=80)
                for b_id, batch in enumerate(t):
                    batch = {k: v.to(self.device_first) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                    with torch.no_grad():
                        self.bart_model.eval()
                        latent = self.bart_model.get_encoder()(input_ids = batch['input_ids'], attention_mask = batch['attention_mask']).last_hidden_state
                        if self.cfg.diffusion.normalize_latent:
                            if self.epoch == 0 and b_id == 0:
                                latent_vecs = torch.cat([latent[i][:torch.sum(batch['attention_mask'][i])] for i in range(latent.shape[0])], dim=0)
                                self.diffusion.module.latent_mean = torch.mean(latent_vecs, dim=0)
                                self.ema.ema_model.latent_mean = self.diffusion.module.latent_mean
                                self.diffusion.module.latent_scale = torch.std(latent_vecs-self.diffusion.module.latent_mean, unbiased=False)
                                self.ema.ema_model.latent_scale = self.diffusion.module.latent_scale

                            latent = self.diffusion.module.normalize_latent(latent)
                    # get mask
                    mask = batch['attention_mask'].bool()
                    # keyword processing
                    # TODO (WEAK): move to dataloader
                    if self.is_keyword:
                        # pad to the longest tokenized keyword length
                        kw_ids_ls = [torch.tensor(self.keywords_list[batch['audio_idx'][i].to(torch.int)]['input_ids']) for i in range(len(batch['input_ids']))]
                        kw_ids = torch.nn.utils.rnn.pad_sequence(kw_ids_ls, batch_first=True, padding_value=self.tokenizer.pad_token_id).to(self.device_first)
                        kw_mask = torch.nn.utils.rnn.pad_sequence([torch.tensor(self.keywords_list[batch['audio_idx'][i].to(torch.int)]['attention_mask']) for i in range(len(batch['attention_mask']))], batch_first=True, padding_value=0).to(self.device_first).to(torch.bool)
                    else:
                        kw_ids = None
                        kw_mask = None
                    # get loss and grad
                    src = {
                        'waveform': batch['waveform'],
                        'audio_features': batch['audio_features'] if 'audio_features' in batch.keys() else None,
                        'audio_features_attn_mask': batch['audio_features_attn_mask'] if 'audio_features_attn_mask' in batch.keys() else None
                    }
                    if self.is_keyword:
                        src['kw_latent'] = self.bart_model.get_encoder()(input_ids=kw_ids, attention_mask=kw_mask).last_hidden_state
                        src['kw_mask'] = kw_mask
                    # print(f'[DBG]kw_ids: {kw_ids}, kw_mask: {kw_mask}')
                    loss, model_pred = self.diffusion(latent, mask,
                                            src=src, class_id=None)
                    if loss.shape != torch.Size([]):
                        loss = loss.mean()

                    # audio_encoder_alignment
                    if self.audio_encoder_alignment:
                        encoded_audio = model_pred.encoded_audio
                        # ntxent loss between encoded_audio and text latent
                        # the audio features an audio head
                        # do masked mean of latent. mask is attention mask (1 for available)
                        if self.cfg.model.audio_encoder_alignment_text_emb_type == 'bartmean':
                            latent_mean_target_denom = torch.sum(mask, dim=1, keepdim=True) # B 1
                            latent_mean_target = torch.sum(latent*mask.unsqueeze(-1).int(), dim=1)/latent_mean_target_denom # B C
                        elif self.cfg.model.audio_encoder_alignment_text_emb_type == 'sbert':
                            latent_mean_target = batch['text_features']
                        else:
                            raise NotImplementedError(f"audio_encoder_alignment_text_emb_type {self.cfg.model.audio_encoder_alignment_text_emb_type} not implemented.")
                        loss_audio_encoder_alignment = self.cfg.model.audio_encoder_alignment_loss_weight*self.alignment_loss(encoded_audio, latent_mean_target, batch['audio_idx'])
                        loss += loss_audio_encoder_alignment
                    loss = loss / self.gradient_accumulate_every
                    train_loss_sum += loss.item()
                    loss.backward()
                    grad_norm = compute_grad_norm(self.diffusion.module.parameters())
                    log_dict = {'loss_batch':loss,'grad_norm':grad_norm, 'lr':self.lr_scheduler.get_last_lr()[0], 'batch_step':b_id+self.epoch*len(self.train_dataloader)}
                    wandb.log(log_dict)
                    if (b_id + 1) % self.gradient_accumulate_every == 0 or (b_id + 1) == len(self.train_dataloader):
                        self.optimizer.step()
                        self.lr_scheduler.step()
                        self.optimizer.zero_grad()
                        self.ema.to(self.device_first)
                        self.ema.update()

                self.diffusion.eval()
                self.ema.ema_model.eval()
                self.bart_model.eval()
                eval_loss_sum = {'val':0, 'test':0} # for logging
                eval_loss_ema_sum = {'val':0, 'test':0} # for logging
                eval_loss_dataloader = {'val':self.val_dataloader, 'test':self.test_dataloader}
                for eval_sp in ['val', 'test']:
                    with torch.no_grad():
                        t_eval = tqdm(eval_loss_dataloader[eval_sp], ncols=80)
                        for b_id, batch in enumerate(t_eval):
                            batch = {k: v.to(self.device_first) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                            with torch.no_grad():
                                # get latent from BART encoder
                                latent = self.bart_model.get_encoder()(input_ids=batch['input_ids'], attention_mask=batch[
                                    'attention_mask']).last_hidden_state
                                # normalize latent space
                                if self.cfg.diffusion.normalize_latent:
                                    latent = self.diffusion.module.normalize_latent(latent)
                            # get mask
                            mask = batch['attention_mask'].bool()
                            # keyword processing
                            # TODO (WEAK): move to dataloader
                            if self.is_keyword:
                                kw_ids = torch.nn.utils.rnn.pad_sequence([torch.tensor(self.keywords_list[batch['audio_idx'][i].to(torch.int)]['input_ids']) for i in range(len(batch['input_ids']))], batch_first=True, padding_value=self.tokenizer.pad_token_id).to(self.device_first)
                                kw_mask = torch.nn.utils.rnn.pad_sequence([torch.tensor(self.keywords_list[batch['audio_idx'][i].to(torch.int)]['attention_mask']) for i in range(len(batch['attention_mask']))], batch_first=True, padding_value=0).to(self.device_first).to(torch.bool)
                            else:
                                kw_ids = None
                                kw_mask = None
                            # get loss
                            src = {
                                'waveform': batch['waveform'],
                                'audio_features': batch['audio_features'] if 'audio_features' in batch.keys() else None,
                                'audio_features_attn_mask': batch['audio_features_attn_mask'] if 'audio_features_attn_mask' in batch.keys() else None
                            }
                            if self.is_keyword:
                                src['kw_latent'] = self.bart_model.get_encoder()(input_ids=kw_ids, attention_mask=kw_mask).last_hidden_state
                                src['kw_mask'] = kw_mask
                            loss, model_pred = self.diffusion(latent, mask,
                                                    src=src, class_id=None)
                            if loss.shape != torch.Size([]):
                                loss = loss.mean()
                            loss = loss / self.gradient_accumulate_every
                            eval_loss_sum[eval_sp] += loss.item()
                            # get loss from EMA model
                            loss, _ = self.ema(latent, mask, src=src, class_id=None)
                            if loss.shape != torch.Size([]):
                                loss = loss.mean()
                            loss = loss / self.gradient_accumulate_every
                            eval_loss_ema_sum[eval_sp] += loss.item()
                            # if self.cfg.system.debug:
                            #     break

                self.diffusion.train()
                self.ema.ema_model.train()
                result_metric_tracking = {'epoch': self.epoch, 'train': {'loss':train_loss_sum / len(self.train_dataloader)}, 'val': {'loss':eval_loss_sum['val'] / len(self.val_dataloader),'ema_loss':eval_loss_ema_sum['val'] / len(self.val_dataloader)}, 'test': {'loss':eval_loss_sum['test'] / len(self.test_dataloader),'ema_loss':eval_loss_ema_sum['test'] / len(self.val_dataloader)}}
                self.logger.info(f"Epoch[{self.epoch}/{self.max_epochs}] finished. Train Loss:{result_metric_tracking['train']['loss']}")
                result_metric_val_tracking_str = '||'.join([f'{k}: {v:.4f}' for k,v in result_metric_tracking['val'].items()])
                self.logger.info(f"Val  Metrics: {result_metric_val_tracking_str}")
                result_metric_test_tracking_str = '||'.join([f'{k}: {v:.4f}' for k,v in result_metric_tracking['test'].items()])
                self.logger.info(f"Test Metrics: {result_metric_test_tracking_str}")
                wandb.log(result_metric_tracking)
                result_metric_tracking_all[self.epoch] = result_metric_tracking
                improved_metric = []
                for metric, mode in zip(self.cfg.train.monitor_metrics,self.cfg.train.monitor_modes):
                    if self.best_tracker['val'][metric]['perf'] is None: # first time
                        self.best_tracker['val'][metric]['perf'] = result_metric_tracking['val'][metric]
                        self.best_tracker['val'][metric]['epoch'] = self.epoch
                        self.best_tracker['test'][metric]['perf'] = result_metric_tracking['test'][metric]
                        self.best_tracker['test'][metric]['epoch'] = self.epoch
                        improved_metric.append(metric)
                    if mode == 'max':
                        if result_metric_tracking['val'][metric] > self.best_tracker['val'][metric]['perf']:
                            self.best_tracker['val'][metric]['perf'] = result_metric_tracking['val'][metric]
                            self.best_tracker['val'][metric]['epoch'] = self.epoch
                            improved_metric.append(metric)
                        if result_metric_tracking['test'][metric] > self.best_tracker['test'][metric]['perf']:
                            self.best_tracker['test'][metric]['perf'] = result_metric_tracking['test'][metric]
                            self.best_tracker['test'][metric]['epoch'] = self.epoch
                    elif mode == 'min':
                        if result_metric_tracking['val'][metric] < self.best_tracker['val'][metric]['perf']:
                            self.best_tracker['val'][metric]['perf'] = result_metric_tracking['val'][metric]
                            self.best_tracker['val'][metric]['epoch'] = self.epoch
                            improved_metric.append(metric)
                        if result_metric_tracking['test'][metric] < self.best_tracker['test'][metric]['perf']:
                            self.best_tracker['test'][metric]['perf'] = result_metric_tracking['test'][metric]
                            self.best_tracker['test'][metric]['epoch'] = self.epoch
                    else:
                        raise NotImplementedError(f"Mode {mode} not implemented.")
                    self.logger.info(f"For {metric}, current best Val Epoch: {self.best_tracker['val'][metric]['epoch']}, Perf: {self.best_tracker['val'][metric]['perf']}; Current best Test Epoch: {self.best_tracker['test'][metric]['epoch']}, Perf: {self.best_tracker['test'][metric]['perf']}")
                # save checkpoint
                self.save_checkpoint(best_tracker=self.best_tracker, improved_metric_val=improved_metric)
                # early-stopping
                if self.cfg.train.early_stop_patience>0:
                    if len(improved_metric) == 0:
                        self.no_improvement_count += 1
                    if self.no_improvement_count > self.cfg.train.early_stop_patience:
                        self.logger.info(f"No improvement in {self.cfg.train.early_stop_patience} epochs. Early stopping.")
                        break
                # increment epoch
                self.epoch += 1
                # save metric dict at every epoch
                with open(os.path.join(self.outdir, 'result_metric_all.pkl'), 'wb') as f:
                    pkl.dump(result_metric_tracking_all, f)

        except KeyboardInterrupt:
            self.logger.info('=> User Stop!')
            with open(os.path.join(self.outdir, 'result_metric_all.pkl'), 'wb') as f:
                pkl.dump(result_metric_tracking_all, f)
            self.save_interrputed_checkpoint(b_id=b_id, best_tracker=self.best_tracker)

        for metric in self.cfg.train.monitor_metrics:
            self.logger.info(f"For {metric}, best val epoch: {self.best_tracker['val'][metric]['epoch']}, best val perf: {self.best_tracker['val'][metric]['perf']}")
        with open(os.path.join(self.outdir,'result_metric_all.pkl'),'wb') as f:
            pkl.dump(result_metric_tracking_all,f)
        print('=> Training Finished!')
        return result_metric_tracking_all

    def load(self):
        ckpt_path = self.cfg.system.ckpt_path
        ckpt = torch.load(ckpt_path,map_location=self.device_first)
        # revert to the unparallel model
        self.diffusion.to(self.device_first)
        self.ema.to(self.device_first)
        # make sure optimizer, lr_scheduler parameters on correct device
        self.optimizer = get_adamw_optimizer(self.diffusion.parameters(), lr=self.cfg.optimizer.lr, betas=self.cfg.optimizer.betas, weight_decay=self.cfg.optimizer.weight_decay)
        assert self.train_dataloader is not None, 'train_dataloader is None'
        num_training_steps = math.ceil(len(self.train_dataloader) / self.cfg.train.gradient_accumulate_every) * self.cfg.train.max_epochs 
        self.lr_scheduler = get_scheduler(self.cfg.lr_scheduler.type, optimizer=self.optimizer, num_warmup_steps=self.cfg.lr_scheduler.num_warmup_steps,num_training_steps=num_training_steps)

        self.diffusion.module.load_state_dict(ckpt['model']) # type:ignore
        self.ema.load_state_dict(ckpt['ema'])
        self.epoch = ckpt['epoch']
        self.optimizer.load_state_dict(ckpt['optimizer'])
        self.lr_scheduler.load_state_dict(ckpt['lr_scheduler'])
        self.best_tracker = ckpt['best_tracker']
        # update no_improvement_count
        if self.cfg.train.early_stop_patience>0:
            self.no_improvement_count = self.epoch-max([self.best_tracker['val'][metric]['epoch'] for metric in self.cfg.train.monitor_metrics])
        
        self.logger.info(f'Load ckpt from {ckpt_path}, epoch {self.epoch}')
    
    @torch.no_grad()
    def resume_test(self):
        # delete train dataloader to free up memory
        to_del = self.train_dataloader
        self.train_dataloader = None
        del to_del
        # load inference ckpt
        inference_ckpt_path = self.cfg.system.inference_ckpt_path
        with open(inference_ckpt_path,'rb') as f:
            inference_ckpt = pkl.load(f)
        # get start_batch_id
        start_batch_id = int(inference_ckpt[-1]/self.cfg.data['batch_size'])
        # load checkpoint
        ckpt_path = self.cfg.system.ckpt_path
        ckpt = torch.load(ckpt_path,map_location=self.device_first)
        self.diffusion.to(self.device_first)
        self.ema.to(self.device_first)
        self.diffusion.module.load_state_dict(ckpt['model']) # type:ignore
        self.ema.load_state_dict(ckpt['ema'])
        self.epoch = ckpt['epoch']
        # evaluation
        result_metric_test = self.eval_resume(start_batch_id=start_batch_id,inference_ckpt=inference_ckpt,inference_ckpt_path=inference_ckpt_path,split='test', write_file=f'ep_{self.epoch}')
        result_metric_test_str = '||'.join([f'{k}: {v:.4f}' for k,v in result_metric_test.items()])
        self.logger.info(f'Test at ckpt {ckpt_path}:')
        self.logger.info(f"Epoch: {ckpt['epoch']}")
        self.logger.info(f"Test Metrics: {result_metric_test_str}")
        return result_metric_test

    @torch.no_grad()
    def test(self):
        # delete train dataloader to free up memory
        to_del = self.train_dataloader
        self.train_dataloader = None
        del to_del
        # load checkpoint
        ckpt_path = self.cfg.system.ckpt_path
        ckpt = torch.load(ckpt_path,map_location=self.device_first)
        self.diffusion.to(self.device_first)
        self.ema.to(self.device_first)
        self.diffusion.module.load_state_dict(ckpt['model']) # type:ignore
        self.ema.load_state_dict(ckpt['ema'])
        self.epoch = ckpt['epoch']
        # evaluation
        result_metric_test = self.eval(split='test', write_file=f'ep_{self.epoch}')
        result_metric_test_str = '||'.join([f'{k}: {v:.4f}' for k,v in result_metric_test.items()])
        self.logger.info(f'Test at ckpt {ckpt_path}:')
        self.logger.info(f"Epoch: {ckpt['epoch']}")
        self.logger.info(f"Test Metrics: {result_metric_test_str}")
        return result_metric_test

    @torch.no_grad()
    def eval(self, split, write_file:Union[bool,str]=False):
        # supress errors
        assert self.tokenizer.pad_token_id is not None
        assert type(self.ema.ema_model) == GaussianDiffusion
        # make word_list_confiner
        json_word_ids_blacklist = json.load(open(self.cfg.eval.word_ids_blacklist_file))
        if self.cfg.eval.confine_to_word_list:
            self.logger.info('Confining to word list.')
        word_list_confiner = LogitsProcessorList([WordListConfiner(word_blacklist=json_word_ids_blacklist,confine_to_word_list=self.cfg.eval.confine_to_word_list)])
        self.diffusion.eval()
        self.ema.ema_model.eval()
        split_to_loader = {'val':self.val_dataloader_return_dict, 'test':self.test_dataloader_return_dict}
        if self.is_keyword:
            split_to_kw_list = {'train':self.keywords_list, 'val':self.val_keywords, 'test':self.test_keywords}
            kw_list = split_to_kw_list[split]
        else:
            kw_list = None
        data_loader = split_to_loader[split]
        t = tqdm(data_loader, desc=f'Eval {split}', ncols=80)
        start_time = time.time()
        for b_id, eval_batch in enumerate(t):
            texts_lists_batch = {k:[] for k, _ in self.cfg.eval.sample_kwargs.items()}
            file_names_batch = []
            ref_captions_dict_batch = []

            batch_model_output_latents = []
            batch_model_output_masks = []
            src = {'waveform': eval_batch['waveform'],# time first, (time,batch,feature)
                   'audio_features': eval_batch['audio_features'] if 'audio_features' in eval_batch.keys() else None,
                   'audio_features_attn_mask': eval_batch['audio_features_attn_mask'] if 'audio_features_attn_mask' in eval_batch.keys() else None} 

            if self.is_keyword:
                assert kw_list is not None
                kw_ids = torch.nn.utils.rnn.pad_sequence([torch.tensor(kw_list[eval_batch['audio_idx'][i].to(torch.int)]['input_ids']) for i in range(len(eval_batch['input_ids']))], batch_first=True, padding_value=self.tokenizer.pad_token_id).to(self.device_first)
                kw_mask = torch.nn.utils.rnn.pad_sequence([torch.tensor(kw_list[eval_batch['audio_idx'][i].to(torch.int)]['attention_mask']) for i in range(len(eval_batch['attention_mask']))], batch_first=True, padding_value=0).to(self.device_first).to(torch.bool)
                src['kw_latent'] = self.bart_model.get_encoder()(input_ids=kw_ids, attention_mask=kw_mask).last_hidden_state
                src['kw_mask'] = kw_mask
            # put src to device
            src = {k: v.to(self.device_first) if isinstance(v, torch.Tensor) else v for k, v in src.items()}
            # sample latent
            for repeat_id in range(self.cfg.eval['samples_per_audio']):
                model_outputs = self.ema.ema_model.sample(src=src,
                                                          length=self.length_categorical.sample((src['waveform'].size(0),)),# type: ignore
                                                          batch_size=src['waveform'].size(0),
                                                          langevin_fn=self.langevin_fn)
                batch_model_output_latents.append(model_outputs[0].unsqueeze(1).to('cpu')) # (batch, 1, length, dim)# type: ignore
                batch_model_output_masks.append(model_outputs[1].unsqueeze(1).to('cpu')) # ((batch, 1, length)# type: ignore
            
            batch_model_output_latents = torch.cat(batch_model_output_latents, dim=1) # (batch, samples_per_audio, length, dim)
            batch_model_output_masks = torch.cat(batch_model_output_masks, dim=1) # (batch, samples_per_audio, length)
            batch_model_output_latents = batch_model_output_latents.view(-1, batch_model_output_latents.size(-2), batch_model_output_latents.size(-1))  # (batch, samples_per_audio, length, dim) -> (batch*samples_per_audio, length, dim)
            batch_model_output_masks = batch_model_output_masks.view(-1,batch_model_output_masks.size(-1))  # (batch, samples_per_audio, length) -> (batch*samples_per_audio, length)
            # chunked decoding
            # decode, 1) beam, 2) neucleus
            # if self.cfg.system.debug:
            #     # save the first batch
            #     torch.save(batch_model_output_latents, os.path.join(self.outdir, f'batch_model_output_latents_{split}_{self.epoch}.pth'))
            #     torch.save(batch_model_output_masks, os.path.join(self.outdir, f'batch_model_output_masks_{split}_{self.epoch}.pth'))
            for k, kwargs in self.cfg.eval.sample_kwargs.items():
                temp_texts_list = []
                for decode_bid in range(math.ceil(batch_model_output_latents.size(0)/self.cfg.data['batch_size_eval'])):
                    latents = batch_model_output_latents[decode_bid*self.cfg.data['batch_size_eval']:(decode_bid+1)*self.cfg.data['batch_size_eval']]
                    mask = batch_model_output_masks[decode_bid*self.cfg.data['batch_size_eval']:(decode_bid+1)*self.cfg.data['batch_size_eval']]
                    latents, mask = latents.to(self.device_first), mask.to(self.device_first)
                    if self.cfg.diffusion.normalize_latent:
                        latents = self.ema.ema_model.unnormalize_latent(latents)
                    encoder_output = BaseModelOutput(last_hidden_state=latents.clone()) # type: ignore
                    sample_ids = self.bart_model.generate(encoder_outputs=encoder_output, attention_mask=mask.clone(), logits_processor=word_list_confiner,**kwargs)
                    texts_list = [self.tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in sample_ids]
                    texts_list = [text.strip() for text in texts_list if len(text.strip())>0] # shape: (batch)
                    # accumulate
                    temp_texts_list.extend(texts_list) # (batch*samples_per_audio)
                # split BY audio
                temp_texts_list = [temp_texts_list[i*self.cfg.eval.samples_per_audio:(i+1)*self.cfg.eval.samples_per_audio] for i in range(src['waveform'].size(0))]
                texts_lists_batch[k].extend(temp_texts_list)
            ref_captions_dict_batch.extend(eval_batch['caption'])
            file_names_batch.extend(eval_batch['audio_name']) # TODO: change to mbr decoding, select best with mbleu
            self.write_output_file(texts_lists_batch, ref_captions_dict_batch, file_names_batch, write_file=f'{split}_{write_file}' if write_file else False)
            print(f'Current batch {b_id} finished. Time elapsed: {time.time()-start_time:.2f}s, the save path is {self.outdir}')
            if b_id == 0:
                texts_lists_all, ref_captions_dict_all, file_names_all = texts_lists_batch, ref_captions_dict_batch, file_names_batch
            else:
                for k in texts_lists_all.keys(): # type: ignore
                    texts_lists_all[k].extend(texts_lists_batch[k]) # type: ignore
                ref_captions_dict_all.extend(ref_captions_dict_batch) # type: ignore
                file_names_all.extend(file_names_batch) # type: ignore
            # if self.cfg.system.debug:
            #     break
            eval_time = time.time() - start_time
            if write_file:
                with open(os.path.join(self.outdir, f'text_ref_fname_{split}_{write_file}.pkl'), 'wb') as f:
                    pkl.dump((texts_lists_all, ref_captions_dict_all, file_names_all, (b_id+1)*self.cfg.data['batch_size_eval']), f) # type: ignore
        metrics = {k:{} for k, _ in self.cfg.eval.sample_kwargs.items()}
        for k in self.cfg.eval.sample_kwargs.keys():
            print(f'For Generation method {k}:')
            metrics[k].update(evaluate_metrics_diversity(texts_lists_all[k],candidate_size=5)) # type: ignore
            if k == list(self.cfg.eval.sample_kwargs.keys())[0]:
                for metric, values in metrics[k].items():
                    print(f'{k}: {metric:<7s}: {values["score"]:7.4f}')

        self.diffusion.train()
        self.ema.ema_model.train() # cancel the side effect
        metrics_simple = convert_metrics(metrics)
        if write_file:
            with open(os.path.join(self.outdir, f'metrics_{split}_{write_file}.pkl'), 'wb') as f:
                pkl.dump(metrics, f)
        return metrics_simple

    def eval_resume(self,start_batch_id, inference_ckpt, inference_ckpt_path, split, write_file):
        # supress errors
        assert self.tokenizer.pad_token_id is not None
        assert type(self.ema.ema_model) == GaussianDiffusion
        # make word_list_confiner
        json_word_ids_blacklist = json.load(open(self.cfg.eval.word_ids_blacklist_file))
        if self.cfg.eval.confine_to_word_list:
            self.logger.info('Confining to word list.')
        word_list_confiner = LogitsProcessorList([WordListConfiner(word_blacklist=json_word_ids_blacklist,confine_to_word_list=self.cfg.eval.confine_to_word_list)])
        self.diffusion.eval()
        self.ema.ema_model.eval()
        split_to_loader = {'val':self.val_dataloader_return_dict, 'test':self.test_dataloader_return_dict}
        if self.is_keyword:
            split_to_kw_list = {'train':self.keywords_list, 'val':self.val_keywords, 'test':self.test_keywords}
            kw_list = split_to_kw_list[split]
        else:
            kw_list = None
        data_loader = split_to_loader[split]
        t = tqdm(data_loader, desc=f'Eval {split}', ncols=80)
        start_time = time.time()
        for b_id, eval_batch in enumerate(t):
            # skip until start_batch_id
            if b_id < start_batch_id:
                print(f'Skipping batch {b_id}')
                continue
            texts_lists_batch = {k:[] for k, _ in self.cfg.eval.sample_kwargs.items()}
            file_names_batch = []
            ref_captions_dict_batch = []

            batch_model_output_latents = []
            batch_model_output_masks = []
            src = {'waveform': eval_batch['waveform'], # time first, (time,batch,feature)
                   'audio_features': eval_batch['audio_features'] if 'audio_features' in eval_batch.keys() else None,
                   'audio_features_attn_mask': eval_batch['audio_features_attn_mask'] if 'audio_features_attn_mask' in eval_batch.keys() else None} 

            if self.is_keyword:
                assert kw_list is not None
                kw_ids = torch.nn.utils.rnn.pad_sequence([torch.tensor(kw_list[eval_batch['audio_idx'][i].to(torch.int)]['input_ids']) for i in range(len(eval_batch['input_ids']))], batch_first=True, padding_value=self.tokenizer.pad_token_id).to(self.device_first)
                kw_mask = torch.nn.utils.rnn.pad_sequence([torch.tensor(kw_list[eval_batch['audio_idx'][i].to(torch.int)]['attention_mask']) for i in range(len(eval_batch['attention_mask']))], batch_first=True, padding_value=0).to(self.device_first).to(torch.bool)
                src['kw_latent'] = self.bart_model.get_encoder()(input_ids=kw_ids, attention_mask=kw_mask).last_hidden_state
                src['kw_mask'] = kw_mask
            # put src to device
            src = {k: v.to(self.device_first) if isinstance(v, torch.Tensor) else v for k, v in src.items()}
            # sample latent
            for repeat_id in range(self.cfg.eval['samples_per_audio']):
                model_outputs = self.ema.ema_model.sample(src=src,
                                                          length=self.length_categorical.sample((src['waveform'].size(0),)),# type: ignore
                                                          batch_size=src['waveform'].size(0),
                                                          langevin_fn=self.langevin_fn)
                batch_model_output_latents.append(model_outputs[0].unsqueeze(1).to('cpu')) # (batch, 1, length, dim)# type: ignore
                batch_model_output_masks.append(model_outputs[1].unsqueeze(1).to('cpu')) # ((batch, 1, length)# type: ignore
            
            batch_model_output_latents = torch.cat(batch_model_output_latents, dim=1) # (batch, samples_per_audio, length, dim)
            batch_model_output_masks = torch.cat(batch_model_output_masks, dim=1) # (batch, samples_per_audio, length)
            batch_model_output_latents = batch_model_output_latents.view(-1, batch_model_output_latents.size(-2), batch_model_output_latents.size(-1))  # (batch, samples_per_audio, length, dim) -> (batch*samples_per_audio, length, dim)
            batch_model_output_masks = batch_model_output_masks.view(-1,batch_model_output_masks.size(-1))  # (batch, samples_per_audio, length) -> (batch*samples_per_audio, length)
            # chunked decoding
            # decode, 1) beam, 2) neucleus
            # if self.cfg.system.debug:
            #     # save the first batch
            #     torch.save(batch_model_output_latents, os.path.join(self.outdir, f'batch_model_output_latents_{split}_{self.epoch}.pth'))
            #     torch.save(batch_model_output_masks, os.path.join(self.outdir, f'batch_model_output_masks_{split}_{self.epoch}.pth'))
            for k, kwargs in self.cfg.eval.sample_kwargs.items():
                temp_texts_list = []
                for decode_bid in range(math.ceil(batch_model_output_latents.size(0)/self.cfg.data['batch_size_eval'])):
                    latents = batch_model_output_latents[decode_bid*self.cfg.data['batch_size_eval']:(decode_bid+1)*self.cfg.data['batch_size_eval']]
                    mask = batch_model_output_masks[decode_bid*self.cfg.data['batch_size_eval']:(decode_bid+1)*self.cfg.data['batch_size_eval']]
                    latents, mask = latents.to(self.device_first), mask.to(self.device_first)
                    if self.cfg.diffusion.normalize_latent:
                        latents = self.ema.ema_model.unnormalize_latent(latents)
                    encoder_output = BaseModelOutput(last_hidden_state=latents.clone()) # type: ignore
                    sample_ids = self.bart_model.generate(encoder_outputs=encoder_output, attention_mask=mask.clone(), logits_processor=word_list_confiner,**kwargs)
                    texts_list = [self.tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in sample_ids]
                    texts_list = [text.strip() for text in texts_list] # shape: (batch)
                    # accumulate
                    temp_texts_list.extend(texts_list) # (batch*samples_per_audio)
                # split BY audio
                temp_texts_list = [temp_texts_list[i*self.cfg.eval.samples_per_audio:(i+1)*self.cfg.eval.samples_per_audio] for i in range(src['waveform'].size(0))]
                texts_lists_batch[k].extend(temp_texts_list)
            ref_captions_dict_batch.extend(eval_batch['caption'])
            file_names_batch.extend(eval_batch['audio_name']) # TODO: change to mbr decoding, select best with mbleu
            self.write_output_file(texts_lists_batch, ref_captions_dict_batch, file_names_batch, write_file=f'{split}_{write_file}' if write_file else False)
            print(f'Current batch {b_id} finished. Time elapsed: {time.time()-start_time:.2f}s, the save path is {self.outdir}')
            # restore texts_lists_all, ref_captions_dict_all, file_names_all from inference_ckpt
            if b_id == start_batch_id:
                texts_lists_all, ref_captions_dict_all, file_names_all = inference_ckpt[0], inference_ckpt[1], inference_ckpt[2]
            
            for k in texts_lists_all.keys(): # type: ignore
                texts_lists_all[k].extend(texts_lists_batch[k]) # type: ignore
            ref_captions_dict_all.extend(ref_captions_dict_batch) # type: ignore
            file_names_all.extend(file_names_batch) # type: ignore
            # if self.cfg.system.debug:
            #     break
            eval_time = time.time() - start_time
            if write_file:
                with open(os.path.join(self.outdir, f'text_ref_fname_{split}_{write_file}.pkl'), 'wb') as f:
                    pkl.dump((texts_lists_all, ref_captions_dict_all, file_names_all, (b_id+1)*self.cfg.data['batch_size_eval']), f) # type: ignore
        metrics = {k:{} for k, _ in self.cfg.eval.sample_kwargs.items()}
        for k in self.cfg.eval.sample_kwargs.keys():
            print(f'For Generation method {k}:')
            metrics[k].update(evaluate_metrics_diversity(texts_lists_all[k],candidate_size=5)) # type: ignore
            if k == list(self.cfg.eval.sample_kwargs.keys())[0]:
                for metric, values in metrics[k].items():
                    print(f'{k}: {metric:<7s}: {values["score"]:7.4f}')

        self.diffusion.train()
        self.ema.ema_model.train() # cancel the side effect
        metrics_simple = convert_metrics(metrics)
        if write_file:
            with open(os.path.join(self.outdir, f'metrics_{split}_{write_file}.pkl'), 'wb') as f:
                pkl.dump(metrics, f)
        return metrics_simple

    
    def save_checkpoint(self, best_tracker, improved_metric_val=()):
        save_path = os.path.join(self.outdir, f'current_ckpt.pth')
        current_epoch = self.epoch
        self.diffusion.to('cpu') # type: ignore
        self.ema.to('cpu')
        state = {'epoch': current_epoch, 'model': self.diffusion.module.state_dict(), 'ema':self.ema.state_dict(), 'optimizer': self.optimizer.state_dict(), 'lr_scheduler': self.lr_scheduler.state_dict(), 'best_tracker': best_tracker} # type: ignore
        torch.save(state, save_path)
        if (current_epoch+1) % self.cfg.train.save_every == 0:
            save_path = os.path.join(self.outdir, f'ckpt_{current_epoch}.pth')
            torch.save(state, save_path)
        for metric in improved_metric_val:
            save_path = os.path.join(self.outdir, f'best_val_{metric}.pth')
            torch.save(state, save_path)
        self.diffusion.to(self.device_first)
        self.ema.to(self.device_first)
        self.logger.debug('checkpoint saved')

    def save_interrputed_checkpoint(self, b_id, best_tracker):
        save_path = os.path.join(self.outdir, f'interrupt_ckpt.pth')
        state = {'epoch': self.epoch, 'bid': b_id, 'model': self.diffusion.module.state_dict(), 'ema':self.ema.state_dict(), 'optimizer': self.optimizer.state_dict(), 'lr_scheduler': self.lr_scheduler.state_dict(), 'best_tracker': best_tracker}
        torch.save(state, save_path)
        self.logger.debug('checkpoint saved')

    def write_output_file(self, predicted_output:Dict[str,List], ref_captions: List[Dict], file_names, write_file:Union[bool,str]=False):
        """
        write file for result visualization
        """
        caption_field = ['caption_{}'.format(i) for i in range(1, 6)]
        for k in predicted_output.keys():
            for pred_cap, ref_cap_dict, f_name in zip(predicted_output[k], ref_captions, file_names):
                ref_cap_dict.update({'file_name': f_name})
                gt_caps = [ref_cap_dict[cap_ind] for cap_ind in caption_field]
                write_strings = [f'Captions for file {f_name}:']
                write_strings.extend([f'\t Predicted caption_{i}: {pred_cap[i]}' for i in range(len(pred_cap))])
                write_strings.extend([f'\t Original caption_{i}: {gt_caps[i]}' for i in range(len(gt_caps))])
                if write_file:
                    with open(os.path.join(self.outdir, f'result_samplewith_{k}_{write_file}.txt'), 'a') as f:
                        f.write('\n'.join(write_strings))
                        f.write('\n')
        return

def compute_grad_norm(parameters):
    # implementation adapted from https://pytorch.org/docs/stable/_modules/torch/nn/utils/clip_grad.html#clip_grad_norm_
    parameters = [p for p in parameters if p.grad is not None]
    total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), p=2) for p in parameters]), p=2).item() # type: ignore
    return total_norm

def compute_grad_norm_gradlist(grad_lists):
    # implementation adapted from https://pytorch.org/docs/stable/_modules/torch/nn/utils/clip_grad.html#clip_grad_norm_
    total_norm = torch.norm(torch.stack([torch.norm(g.detach(), p=2) for g in grad_lists]), p=2).item() # type: ignore
    return total_norm