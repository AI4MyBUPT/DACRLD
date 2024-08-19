# from accelerate.logging import get_logger
import logging
from .git import git_commit
import time
import wandb
import os

formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s')

def get_proj_logger(name, level=logging.INFO):
    logger = logging.getLogger(name)
    logger.handlers.clear()
    logger.setLevel(level)
    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger


def add_filehandler(logger, filepath):
    fh = logging.FileHandler(filepath)
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)


def set_logger(cfg):
    now = time.strftime("%Y%m%d-%H%M%S")
    aud_net = str(cfg.model.src_encoder.model)
    if cfg.model.src_encoder.frozen:
        aud_net = aud_net  + '_fr'
    else:
        if cfg.model.src_encoder.pretrained:
            aud_net = aud_net  + '_tune'
        else:
            aud_net = aud_net  + '_scratch'
    tag = f'{now}.data_{cfg.data.dataset}_latentTF_{aud_net}_{cfg.model.ver}_lr_{cfg.optimizer.lr}_sche_{cfg.lr_scheduler.type}_warm_{cfg.lr_scheduler.num_warmup_steps}_bs_{cfg.data.batch_size}'
    outdir = os.path.join('output',tag)
    logger_isdisabled = cfg.system.debug or cfg.system.mode == 'test'
    # git commmit
    git_commit('./', timestamp=now, commit_info=tag, debug=logger_isdisabled)
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    # copy config file
    cfg.dump(os.path.join(outdir,'config.py'))
    print(f'Output directory: {outdir}')
    wandb.init(project=cfg.system.name,
               notes=cfg.system.notes,
               config=cfg._cfg_dict,
               mode='disabled' if logger_isdisabled else 'online',
               name=tag)
    logger = get_proj_logger(cfg.system.name)
    outloggerfile = os.path.join(outdir,'log.txt')
    add_filehandler(logger,outloggerfile)
    return logger, outdir
