import torch
from .DiffusionLatentTransformer import DiffusionLatentTransformer

MODELS = {"DiffusionLatentTransformer": DiffusionLatentTransformer,
          }
def build_model(cfg):
    # example dataset
    # additional_dataset_info = {'categories':dataset_metadata['category_count']}
    # cfg.merge_from_dict(additional_dataset_info)
    cfg_copy = cfg.copy()
    type = cfg_copy.type
    model = MODELS[type](cfg_copy)
    return model