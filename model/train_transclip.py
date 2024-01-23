import torch
import torch.cuda.amp as amp

from tqdm import tqdm
from data import build_dataloader

def save_ckpt(model, metrics, save_path:str, data_args):
    save_items={
        "model": model.state_dict(),
        "args": model.args,
        "data_args": data_args,
        "persona_list": model.persona_list,
        "metric": metrics
    }
    torch.save(save_items, save_path)

@torch.no_grad()
def eval_logic(model, eval_ds, train_conf, logger, verboose=False):
    '''
    perform a simple evaluation after one epoch
    '''
    torch.cuda.empty_cache()
    eval_dl = build_dataloader(eval_ds, batch=train_conf["eval_batch_size"], 
                              num_workers=train_conf["eval_num_workers"],
                              dist_training=False, shuffle=False)
    eval_items = len(eval_ds)
    total_corr = 0.0
    total_loss = 0.0

    logger.info("Evaluation begin")
    for i,item in tqdm(enumerate(eval_dl), total = len(eval_dl), desc=f"Eval"):
        # forward
        model.eval()
        loss, num_corr = model.eval_batch(imgs=item["images"],
                                        gt_captions=item["comment"],
                                        candidates=item["candidates"],
                                        personas=item["personality"])
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