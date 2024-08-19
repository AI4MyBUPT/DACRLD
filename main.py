import torch
import multiprocessing as mp
from lib.train_utils import Trainer
from mmcv import Config
import os
from lib.logger import get_proj_logger, add_filehandler
import time
import argparse

def action(cfg):
    if cfg.system.mode == 'train':                
        trainer = Trainer(cfg=cfg)
        # see whether resume training
        if cfg.system.get('ckpt_path',None):
            if cfg.system.resume_train:
                trainer.load()
        trainer.train()
    elif cfg.system.mode == 'test':
        logger = get_proj_logger(cfg.system.name)
        note_eval = cfg.system.get('note_eval',None)
        time_now = time.strftime("%Y%m%d-%H%M%S", time.localtime())
        put_into_dir = f'eval_{time_now}_{note_eval}' if note_eval else f'eval_{time_now}'
        inference_ckpt_path = cfg.system.get('inference_ckpt_path',None)
        if inference_ckpt_path:
            outdir = os.path.split(inference_ckpt_path)[0]
        else:
            outdir = os.path.join(os.path.split(cfg.system.ckpt_path)[0], put_into_dir)
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        outloggerfile = os.path.join(outdir, 'log.txt')
        add_filehandler(logger, outloggerfile)
        cfg.dump(os.path.join(outdir,'config.py'))
        batch_size_eval = cfg.data.get('batch_size_eval',cfg.data.batch_size)
        cfg.data.batch_size = batch_size_eval
        trainer = Trainer(cfg=cfg,outdir=outdir,logger=logger)
        if inference_ckpt_path:
            trainer.resume_test()
        else:
            trainer.test()
    else:
        raise NotImplementedError
    return trainer

def main(args):
    mp.set_start_method('spawn')
    cfg_path = args.config
    cfg = Config.fromfile(cfg_path)
    if cfg.system.mode == 'train':
        trainer = action(cfg)
        # change cfg.system.ckpt_path and cfg.system.mode = 'test'
        cfg.system.ckpt_path = os.path.join(trainer.outdir,'best_val_ema_loss.pth')
        cfg.system.mode = 'test'
        del trainer
        action(cfg)
    elif cfg.system.mode == 'test':
        action(cfg)
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='audio_cap_diffusion')
    parser.add_argument('--config', default='./lib/config.py', help='config file path')
    args = parser.parse_args()
    main(args)