import os
import json
import torch
import logging
import argparse
import torch.cuda.amp as amp

from tqdm import tqdm
from typing import Dict
from torch.optim import AdamW

from model import TransCLIPModel
from model.train_transclip import (train_iter, eval_logic, save_ckpt)
from data import PCapDataset, build_dataloader
from transformers import (get_linear_schedule_with_warmup, 
                          get_constant_schedule_with_warmup)
from utils.metric_logger import TensorboardLogger
from utils.misc import (conf_from_yaml, get_now_str, 
                        neat_print_dict, get_log_str)

logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_cfg", type=str, default="./misc/train_conf.yaml")
    args = parser.parse_args()

    return args

def get_optim_scheduler(params, lr, total_steps:int, warmup_ratio:float, weight_decay: float=0.05):
    '''
    Return AdamW optim and scheduler with warmup
    '''
    optim = AdamW(params=params, lr=lr, weight_decay=weight_decay)
    scheduler = get_constant_schedule_with_warmup(optim, num_warmup_steps=int(warmup_ratio * total_steps))
    # scheduler = get_linear_schedule_with_warmup(optim, num_warmup_steps=int(warmup_ratio * total_steps),
    #                                         num_training_steps = total_steps)
                                                
    return optim, scheduler

def train(model, data_loader, train_args:Dict[str,str], data_args: Dict[str,str], meter):
    # Training args
    epoches = int(train_args["epoches"])
    lr = float(train_args["lr"])
    warmup_ratio = float(train_args["warmup_ratio"])
    log_step = int(train_args.get("logging_steps", 50))

    # training steps can be inferred from len of dl
    step_per_epoch = len(data_loader)
    total_steps = int(step_per_epoch * epoches)
    
    # 1. load optim and scheduler
    optim, scheduler = get_optim_scheduler(model.parameters(), lr, total_steps, warmup_ratio)
    scaler = amp.grad_scaler.GradScaler(enabled=True)

    logging_file = os.path.join(train_args["work_dir"], "log_output.txt")
    logger.info("-------------Training config------------")
    logger.info("Model config:")
    neat_print_dict(model.args)
    logger.info("Training config:")
    neat_print_dict(train_conf)
    # work dir
    work_dir = train_conf["work_dir"]

    with open(logging_file, "w") as f:
        json.dump(model.args, f, indent=4)
        f.write("\n")
        json.dump(train_conf, f, indent=4)
        f.write("\n")

    logger.info("Start training")
    # record current R@1 and best
    cur_metric = 0.0
    best_metric = 0.0

    # 2. real training step
    global_step = 0
    for e in range(1, epoches+1):
        logger.info(f"---------Start epoch {e}-------------")
        for i, item in tqdm(enumerate(data_loader), 
                           total = step_per_epoch, desc=f"Epoch {e}/{epoches}"):
            # forward
            loss, num_corr = train_iter(model, item, optim, scheduler, scaler)
        
            cur_lr = optim.param_groups[0]["lr"]
            if i % log_step == 0:
                log_items = {
                    "Step": f"{(i+1)}/{step_per_epoch}",
                    "lr": f"{cur_lr:.2e}",
                    "loss": f"{loss.item():.5f}",
                    "num_corr": f"{num_corr}/{train_conf['train_batch_size']}"
                }
                tqdm.write(get_log_str(log_items))

            meter.update_metrics({"train": {"loss": loss}})
            meter.update_params({"params": {"lr": cur_lr}})
            meter.get_logs(global_step)

            global_step += 1
            
        logger.info(f"---------Finished epoch {e}-------------")

        # evaluation after batch
        eval_ds = PCapDataset.from_config(data_args, split="val")
        eval_loss, corr_per = eval_logic(model, eval_ds, train_conf, logger)
        meter.update_metrics({"eval": {"loss": eval_loss, "corr_per": corr_per}})

        # save logs to $(work_dir)/log.txt
        with open(logging_file, "a+") as f:
            json.dump({"epoch":e, "loss": eval_loss, "corr_per": corr_per}, f)
            f.write("\n")
        # save ckpt
        cur_metric = corr_per
        if cur_metric >= best_metric:
            # update current
            best_metric = cur_metric
            best_ckpt_path = os.path.join(work_dir, "best.pth")
            logger.info(f"Saving best ckpt to {best_ckpt_path}")
            save_ckpt(model, metrics={"R@1": best_metric}, 
                      data_args=data_args, save_path=best_ckpt_path)

        # save to last.pth
        ckpt_path = os.path.join(work_dir, f"ckpt_{e}.pth")
        logger.info(f"Saving current ckpt to {ckpt_path}")
        save_ckpt(model, metrics={"R@1": cur_metric}, 
                  data_args=data_args, save_path=ckpt_path)
        
        # empty eval cache
        torch.cuda.empty_cache()

    logger.info("-------------Training done-------------")

    return model

def load_model_and_dl(train_conf, model_conf, device):
    model_args = model_conf["model"]
    data_args = model_conf["data"]

    # make dirs
    logger.info("Loading dataset")
    train_ds = PCapDataset.from_config(data_args, split="train")
    train_dl = build_dataloader(train_ds, batch=train_conf["train_batch_size"], 
                              num_workers=train_conf["train_num_workers"],
                              dist_training=False, shuffle=True)
    logger.info("Done loading dataset")

    logger.info(f"Loading model on device: {device}")
    model = TransCLIPModel(train_ds.persona_lists, model_args).to(device)
    logger.info("Model load done")
    
    return model, train_dl


if __name__=="__main__":
    # data loading logic
    logging.basicConfig(format = "%(asctime)s - %(levelname)s - %(name)s - %(message)s",
                        datefmt = "%Y-%m-%d %H:%M:%S", level = logging.INFO)
    args = parse_args()
    train_conf, model_conf = conf_from_yaml(args.train_cfg)

    work_dir_root = train_conf["work_dir"]
    # make work_dir
    os.makedirs(work_dir_root, exist_ok=True)
    
    # find the max run and create a folder for next run
    import re
    pat = re.compile("^run(\d*)$")
    cur_run = 1
    runs = list(filter(lambda x:re.match(pat, x) is not None, os.listdir(work_dir_root)))
    runs.sort(key=lambda x:int(re.findall(pat, x)[0]))

    if len(runs) > 0:
        cur_run = int(re.findall(pat, runs[-1])[0]) + 1
    work_dir = os.path.join(work_dir_root, f"run{cur_run}")
    os.makedirs(work_dir, exist_ok=True)

    output_dir = os.path.join(work_dir, "output")
    os.makedirs(output_dir, exist_ok=True)
    
    train_conf["work_dir"] = work_dir
    # load metric meter
    log_dir = os.path.join(work_dir, "logs")
    meter = TensorboardLogger(
        log_dir=log_dir
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, train_dl = load_model_and_dl(train_conf, model_conf, device)

    # start train
    model = train(model, train_dl, train_conf, model_conf["data"], meter)
    # save
    ckpt_path = os.path.join(output_dir, f"final_model_{get_now_str()}.pth")
    save_ckpt(model, None, ckpt_path, model_conf["data"])
    logger.info(f"Final model saved to {ckpt_path}")
