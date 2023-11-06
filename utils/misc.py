# Copyright (c) 2020 Microsoft Corporation. Licensed under the MIT license.

import errno
import os
import os.path as op
import yaml
import random
import torch
import numpy as np
import torch.distributed as dist

from PIL import Image
from datetime import datetime as dt

def get_now_str(format:str=None):
    if format is None:
        format="%Y%m%d-%H%M%S"
    
    return dt.now().strftime(format)

def get_log_str(log_items):
    '''
    convert metrics to full logging str
    '''
    log_str = f"[{get_now_str()}]  "
    log_str_items = [f"{key}: {val}" for key, val in log_items.items()]
    log_str += (", ".join(log_str_items))
    
    return log_str

def neat_print_dict(config, indent=0, addition=4):
    '''
    neat print dictionary
    '''
    if not isinstance(config, dict):
        raise TypeError("Argument config must be dict, got {}".format(type(config)))
    for k,v in config.items():
        if isinstance(v, dict):
            print("{}'{}': {{".format(" "*indent, k))
            neat_print_dict(v, indent=indent+addition, addition=addition)
            print("{}}}".format(" "*indent))
            continue

        print("{}'{}': {}".format(" "*indent, k, v))

def conf_from_yaml(train_cfg_file:str):
    '''
    load train config and model config from train_conf.yaml

    returns:
    train_cfg, model_cfg containing ``["model", "data"]``
    '''
    with open(train_cfg_file) as fp:
        train_cfg = yaml.load(fp, Loader=yaml.FullLoader)["train"]
    
    # open model_cfg and data_cfg from model.yaml
    with open(train_cfg["model_cfg"]) as fp:
        model_cfg = yaml.load(fp, Loader=yaml.FullLoader)

    return train_cfg, model_cfg


def mkdir(path):
    # if it is the current folder, skip.
    if path == '':
        return
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def set_seed(seed, n_gpu):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(seed)


def find_file_path_in_yaml(fname, root):
    if fname is not None:
        if op.isfile(fname):
            return fname
        elif op.isfile(op.join(root, fname)):
            return op.join(root, fname)
        else:
            raise FileNotFoundError(
                errno.ENOENT, os.strerror(errno.ENOENT), op.join(root, fname)
            )


def get_rank():
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def get_world_size():
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()

def compute_metrics_from_logits(logits, targets):
    """
        recall@k for N candidates

            logits: (batch_size, num_candidates)
            targets: (batch_size, )
    """
    batch_size, num_candidates = logits.shape

    sorted_indices = logits.sort(descending=True)[1]
    targets = targets.tolist()

    recall_k = dict()
    if num_candidates <= 10:
        ks = [1, max(1, round(num_candidates*0.2)), max(1, round(num_candidates*0.5))]
    elif num_candidates <= 100:
        ks = [1, max(1, round(num_candidates*0.1)), max(1, round(num_candidates*0.5))]
    else:
        raise ValueError("num_candidates: {0} is not proper".format(num_candidates))
    for k in ks:
        # sorted_indices[:,:k]: (batch_size, k)
        num_ok = 0
        for tgt, topk in zip(targets, sorted_indices[:,:k].tolist()):
            if tgt in topk:
                num_ok += 1
        recall_k[f'recall@{k}'] = (num_ok/batch_size)

    # MRR
    MRR = 0
    for tgt, topk in zip(targets, sorted_indices.tolist()):
        rank = topk.index(tgt)+1
        MRR += 1/rank
    MRR = MRR/batch_size
    return recall_k, MRR

