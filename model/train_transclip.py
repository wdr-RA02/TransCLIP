import torch
import torch.cuda.amp as amp

from tqdm import tqdm
from data import build_dataloader

def train_iter(model, item, optim, scheduler, scaler):
    model.train()
    with amp.autocast(enabled=True):
        loss, num_corr = model.forward_batch(imgs=item["images"],
                                             captions=item["comment"],
                                             personas=item["personality"],
                                             training=True)
        
    # fp16 varient of loss.backward
    scaler.scale(loss).backward()
    # fp16 varient of optim.step
    scaler.step(optim)
    scaler.update()
    if scheduler is not None:
        scheduler.step()
    model.zero_grad()
    
    return loss, num_corr

def save_ckpt(model, metrics, save_path:str, data_args):
    save_items={
        "model": model.state_dict(),
        "args": model.args,
        "data_args": data_args,
        "persona_list": model.persona_list,
        "metric": metrics
    }
    torch.save(save_items, save_path)

def eval_logic(model, eval_ds, train_conf, logger, verboose=False):
    '''
    perform a simple evaluation after one epoch
    '''
    torch.cuda.empty_cache()
    eval_dl = build_dataloader(eval_ds, batch=train_conf["eval_batch_size"], 
                              num_workers=train_conf["eval_num_workers"],
                              dist_training=False, shuffle=True)
    eval_items = len(eval_ds)
    total_corr = 0.0
    total_loss = 0.0

    logger.info("Evaluation begin")
    for i,item in tqdm(enumerate(eval_dl), total = len(eval_dl), desc=f"Eval"):
        # forward
        model.eval()
        loss, num_corr = model.forward_batch(imgs=item["images"],
                                        captions=item["comment"],
                                        personas=item["personality"],
                                        training=False)
        # backward
        total_corr+=num_corr
        total_loss+=loss.item()

        if verboose:
            tqdm.write(f"Step: {i}/{len(eval_dl)}, loss: {loss.item():.5f}, num_corr: {num_corr}")

    #calculate mean loss
    mean_loss = total_loss/len(eval_dl)
    corr_per = round(100*total_corr/eval_items, 3)
    logger.info(f"Mean eval loss: {mean_loss}")
    logger.info(f"Correct num: {int(total_corr)}/{eval_items} ({corr_per}%)")
    logger.info("Evaluation done")

    return mean_loss, corr_per