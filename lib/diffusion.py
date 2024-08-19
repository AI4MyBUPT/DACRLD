import math
import copy
from pathlib import Path
import random 
from functools import partial
from collections import namedtuple, Counter
from multiprocessing import cpu_count
import os
import numpy as np
import csv
import timeit
from typing import Union

import torch
from torch import nn, einsum
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from torch.optim import AdamW

from einops import rearrange, reduce
from einops.layers.torch import Rearrange

from PIL import Image
from tqdm.auto import tqdm
from ema_pytorch import EMA

from transformers import get_scheduler, AutoTokenizer, PreTrainedTokenizerBase, T5ForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutput
from transformers.models.bart.modeling_bart import BartForConditionalGeneration

from accelerate import Accelerator
import wandb
# partially adapted from https://github.com/lucidrains/denoising-diffusion-pytorch and https://github.com/justinlovelace/latent-diffusion-for-language


ModelPrediction =  namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start', 'encoded_audio'])


def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def identity(t, *args, **kwargs):
    return t

def cycle(dl):
    while True:
        for data in dl:
            yield data

def has_int_squareroot(num):
    return (math.sqrt(num) ** 2) == num

def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr

def l2norm(t):
    return F.normalize(t, dim = -1)

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def linear_beta_schedule(timesteps):
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype = torch.float64)

def cosine_beta_schedule(timesteps, s = 0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype = torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        model,
        *,
        max_seq_len,
        timesteps = 1000,
        sampling_timesteps = None,
        loss_type = 'l1',
        objective = 'pred_noise',
        beta_schedule = 'cosine',
        p2_loss_weight_gamma = 0., # p2 loss weight, from https://arxiv.org/abs/2204.00227 - 0 is equivalent to weight of 1 across time - 1. is recommended
        p2_loss_weight_k = 1,
        ddim_sampling_eta = 1.,
        langevin_need_restore_norm:Union[bool,tuple,str] = False,
    ):
        super().__init__()

        self.diffusion_model = model

        self.latent_dim = self.diffusion_model.latent_dim
        self.self_condition = self.diffusion_model.self_condition
        self.keyword_condition = self.diffusion_model.keyword_condition
        self.keyword_condition_null_prob = self.diffusion_model.keyword_condition_null_prob
        self.save_each_step = False
        self.max_seq_len = max_seq_len
        self.langevin_need_restore_norm = langevin_need_restore_norm
        if self.langevin_need_restore_norm:
            if langevin_need_restore_norm == 'bart_space':
                self.ase_model_mean = None
                self.ase_model_scale = None
            else:
                self.ase_model_mean = torch.tensor(self.langevin_need_restore_norm[0]) # type: ignore
                self.ase_model_scale = torch.tensor(self.langevin_need_restore_norm[1]) # type: ignore

        self.objective = objective

        assert objective in {'pred_noise', 'pred_x0'}, 'objective must be either pred_noise (predict noise) or pred_x0 (predict image start)'

        if beta_schedule == 'linear':
            betas = linear_beta_schedule(timesteps)
        elif beta_schedule == 'cosine':
            betas = cosine_beta_schedule(timesteps)
        else:
            raise ValueError(f'unknown beta schedule {beta_schedule}')

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value = 1.)

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.loss_type = loss_type

        # sampling related parameters

        self.sampling_timesteps = default(sampling_timesteps, timesteps) # default num sampling timesteps to number of timesteps at training

        assert self.sampling_timesteps <= timesteps # type: ignore
        self.ddim_sampling_eta = ddim_sampling_eta

        # helper function to register buffer from float64 to float32

        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))

        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others

        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        register_buffer('posterior_variance', posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

        register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min =1e-20)))
        register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

        # calculate p2 reweighting

        register_buffer('p2_loss_weight', (p2_loss_weight_k + alphas_cumprod / (1 - alphas_cumprod)) ** -p2_loss_weight_gamma)

        register_buffer('latent_mean', torch.tensor([0]*self.latent_dim))
        register_buffer('latent_scale', torch.tensor(1))


    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def predict_noise_from_start(self, x_t, t, x0):
        return (
            (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) / \
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )

    def normalize_latent(self, x_start):
        eps = 1e-5 
                
        return (x_start-self.latent_mean)/(self.latent_scale+eps) # type: ignore
    
    def normalize_latent_explicit(self, x_start, latent_mean, latent_scale):
        eps = 1e-5 
                
        return (x_start-latent_mean)/(latent_scale+eps)
    
    def unnormalize_latent(self, x_start):
        eps = 1e-5 
        
        return x_start*(self.latent_scale+eps)+self.latent_mean # type: ignore
    
    def unnormalize_latent_explicit(self, x_start, latent_mean, latent_scale):
        eps = 1e-5 
        
        return x_start*(latent_scale+eps)+latent_mean # type: ignore

    def diffusion_model_predictions(self, x, mask, src, t, x_self_cond = None, class_id=None):
        model_output_raw = self.diffusion_model(x, mask, src, t, x_self_cond, class_id=class_id)
        # for the case of multiple outputs, assume the first is the prediction
        model_output = model_output_raw[0] if isinstance(model_output_raw, tuple) else model_output_raw
        audio_encoded = model_output_raw[1] if isinstance(model_output_raw, tuple) else None
        if self.objective == 'pred_noise':
            pred_noise = model_output
            x_start = self.predict_start_from_noise(x, t, pred_noise)

        elif self.objective == 'pred_x0':
            x_start = model_output
            pred_noise = self.predict_noise_from_start(x, t, x_start)
        
        return ModelPrediction(pred_noise, x_start, audio_encoded) # type: ignore

    @torch.no_grad()
    def ddim_sample(self, shape, lengths, src, class_id=None):
        """
            shape: B L C
            length: sampled latent sentence length
            class_id: input condition
        """
        batch, device, total_timesteps, sampling_timesteps, eta, objective = shape[0], self.betas.device, self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta, self.objective

        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)   # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps # type: ignore
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:])) # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        latent = torch.randn(shape, device = device) # start from fully random vector# type: ignore
        mask = [[True]*length + [False]*(self.max_seq_len-length) for length in lengths]
        mask = torch.tensor(mask, dtype=torch.bool, device=device)# type: ignore
        if self.save_each_step:
            latent_steps = []
            x_start_steps = []

        x_start = None
        for time, time_next in tqdm(time_pairs, desc = 'sampling loop time step'):
            time_cond = torch.full((batch,), time, device=device, dtype=torch.long)# type: ignore
            self_cond = x_start if self.self_condition else None
            pred_noise, x_start, *_ = self.diffusion_model_predictions(latent, mask, src, time_cond, self_cond, class_id=class_id)

            if time_next < 0:
                latent = x_start
                continue

            # https://nn.labml.ai/diffusion/stable_diffusion/sampler/ddim 

            alpha = self.alphas_cumprod[time] # \alpha_t (cumprod)  # type: ignore
            alpha_next = self.alphas_cumprod[time_next] # \alpha_{t-1} (cumprod)  # type: ignore

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt() # ddim_sigma
            c = (1 - alpha_next - sigma ** 2).sqrt() # sigma for noise term z_t

            noise = torch.randn_like(latent)
            latent = x_start * alpha_next.sqrt() + \
                  c * pred_noise + \
                  sigma * noise
            # latent: latent for x_{t-1}
            # x_start: x_0_hat, can be calculated from x_t and pred_noise
            # pred_noise: \epsilon_\theta(x_t), can be calculated from x_0_hat and x_t
            # noise: fully random noise
                        
            if self.save_each_step:
                latent_steps.append(latent.unsqueeze(1).to('cpu')) # type: ignore # B 1 L C
                x_start_steps.append(x_start.unsqueeze(1).to('cpu')) # type: ignore # B 1 L C

        if self.save_each_step:
            latent_steps = torch.cat(latent_steps, dim=1) # type: ignore
            x_start_steps = torch.cat(x_start_steps, dim=1) # type: ignore
            return (latent, mask), (latent_steps, x_start_steps) # type: ignore
        else:
            return (latent, mask)


    @torch.no_grad()
    def ddim_sample_langevin(self, shape, lengths, src, class_id=None, langevin_fn=None):
        """
            shape: B L C
            length: sampled latent sentence length
            class_id: input condition
        """
        batch, device, total_timesteps, sampling_timesteps, eta, objective = shape[0], self.betas.device, self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta, self.objective

        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)   # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps # type: ignore
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:])) # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        latent = torch.randn(shape, device = device) # start from fully random vector # type: ignore
        mask = [[True]*length + [False]*(self.max_seq_len-length) for length in lengths]
        mask = torch.tensor(mask, dtype=torch.bool, device=device)# type: ignore
        if self.save_each_step:
            latent_steps = []
            x_start_steps = []

        x_start = None
        for time, time_next in tqdm(time_pairs, desc = 'sampling loop time step'):
            time_cond = torch.full((batch,), time, device=device, dtype=torch.long)# type: ignore
            prev_latent = latent
            self_cond = x_start if self.self_condition else None
            pred_noise, x_start, *_ = self.diffusion_model_predictions(latent, mask, src, time_cond, self_cond, class_id=class_id)

            if time_next < 0:
                latent = x_start
                continue

            # https://nn.labml.ai/diffusion/stable_diffusion/sampler/ddim 

            alpha = self.alphas_cumprod[time] # \alpha_t (cumprod)  # type: ignore
            alpha_next = self.alphas_cumprod[time_next] # \alpha_{t-1} (cumprod)  # type: ignore

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt() # ddim_sigma
            c = (1 - alpha_next - sigma ** 2).sqrt() # sigma for noise term z_t

            noise = torch.randn_like(latent)
            latent = x_start * alpha_next.sqrt() + \
                c * pred_noise + \
                sigma * noise
            # latent: latent for x_{t-1}
            # x_start: x_0_hat, can be calculated from x_t and pred_noise
            # pred_noise: \epsilon_\theta(x_t), can be calculated from x_0_hat and x_t
            # noise: fully random noise
            
            # add langevin support
            if langevin_fn:
                if self.langevin_need_restore_norm:
                    # current model space -> bart space -> langevin space 
                    latent = self.unnormalize_latent(latent)
                    prev_latent = self.unnormalize_latent(prev_latent)
                    if self.langevin_need_restore_norm != 'bart_space':
                        latent = self.normalize_latent_explicit(latent, self.ase_model_mean, self.ase_model_scale)
                        prev_latent = self.normalize_latent_explicit(prev_latent, self.ase_model_mean, self.ase_model_scale)
                latent = langevin_fn(sample=latent,attn_masks=mask, audio_src=src,mean=x_start,sigma=sigma,alpha=alpha,time=time,time_next=time_next,prev_latent=prev_latent)
                if self.langevin_need_restore_norm:
                    # langevin space -> bart space -> current model space
                    if self.langevin_need_restore_norm != 'bart_space':
                        latent = self.unnormalize_latent_explicit(latent, self.ase_model_mean, self.ase_model_scale)
                        prev_latent = self.unnormalize_latent_explicit(prev_latent, self.ase_model_mean, self.ase_model_scale)
                    latent = self.normalize_latent(latent)
                    prev_latent = self.normalize_latent(prev_latent)

            if self.save_each_step:
                latent_steps.append(latent.unsqueeze(1).to('cpu')) # type: ignore # B 1 L C
                x_start_steps.append(x_start.unsqueeze(1).to('cpu')) # type: ignore # B 1 L C

        if self.save_each_step:
            latent_steps = torch.cat(latent_steps, dim=1) # type: ignore
            x_start_steps = torch.cat(x_start_steps, dim=1) # type: ignore
            return (latent, mask), (latent_steps, x_start_steps) # type: ignore
        else:
            return (latent, mask)

    
    @torch.no_grad()
    def sample(self, batch_size, length, src, class_id=None, langevin_fn=None):
        max_seq_len, latent_dim = self.max_seq_len, self.latent_dim
        sample_fn = self.ddim_sample if not langevin_fn else partial(self.ddim_sample_langevin, langevin_fn=langevin_fn)
        return sample_fn((batch_size, max_seq_len, latent_dim), length, src, class_id)

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    @property
    def loss_fn(self):
        if self.loss_type == 'l1':
            return F.l1_loss
        elif self.loss_type == 'l2':
            return F.mse_loss
        elif self.loss_type == 'smooth_l1':
            return F.smooth_l1_loss
        else:
            raise ValueError(f'invalid loss type {self.loss_type}')

    def p_losses(self, x_start, mask, src, t, class_id, noise = None):
        b, l, d = x_start.shape
        noise = default(noise, lambda: torch.randn_like(x_start))

        x = self.q_sample(x_start = x_start, t = t, noise = noise)

        x_self_cond = None
            
        if self.self_condition and random.random() < 0.5:
            with torch.no_grad():
                x_self_cond = self.diffusion_model_predictions(x, mask, src, t, class_id=class_id).pred_x_start.detach()

        predictions = self.diffusion_model_predictions(x, mask, src, t, x_self_cond, class_id=class_id)
                
        loss = self.loss_fn(predictions.pred_x_start, x_start, reduction = 'none')
        loss = rearrange([reduce(loss[i][:torch.sum(mask[i])], 'l d -> 1', 'mean') for i in range(x_start.shape[0])], 'b 1 -> b 1')

        return loss, predictions

    def forward(self, txt_latent, mask, src, class_id, *args, **kwargs):
        b, l, d, device, max_seq_len, = *txt_latent.shape, txt_latent.device, self.max_seq_len
        assert l == max_seq_len, f'length must be {self.max_seq_len}'
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()
        # handle keyword condition
        if self.keyword_condition:
            if random.random() < self.keyword_condition_null_prob:
                src['kw_latent'] = None
                src['kw_mask'] = None
        return self.p_losses(txt_latent, mask, src, t, class_id, *args, **kwargs)

