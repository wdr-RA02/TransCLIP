import torch
import logging
import argparse

from tqdm import tqdm
from data import PCapDataset, build_dataloader
from model.transclip import TransCLIPModel
from utils.misc import neat_print_dict

logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=2)
    #
    parser.add_argument("--cand_source", choices=["additional", "candidates"])
    parser.set_defaults(cand_source="additional")

    args = parser.parse_args()

    return args

def load_from_ckpt(ckpt_path):
    logger.info(f"Loading ckpt from {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")

    logger.info("Model config:")
    neat_print_dict(ckpt["args"])
    logger.info("Model config ends")

    model = TransCLIPModel(
        args=ckpt["args"],
        persona_list=ckpt["persona_list"]
    )
    logger.info("Loading state dict to model")
    logger.info(model.load_state_dict(ckpt["model"]))

    return model, ckpt["data_args"]

def get_candidates(item, source):
    assert source in ["additional", "candidates"]
    if source=="additional":
        candidates = [(orig, *add) for orig, add in zip(item["comment"], item["additional_comments"])]
    else:
        candidates = item["candidates"]
    
    return candidates

@torch.no_grad()
def test_logic(model, test_dataloader, cand_source):
    torch.cuda.empty_cache()
    recall_1 = 0
    loss_avg = 0.0
    logger.info(f"Candidate source: {cand_source}")

    for item in tqdm(test_dataloader, total=len(test_dataloader)):
        candidates = get_candidates(item, cand_source)
        loss, recall = model.eval_batch(item["images"], 
                                        item["personality"], 
                                        item["comment"], 
                                        candidates)
        # print(recall)
        recall_1 += recall
        loss_avg += loss.item()

    return loss_avg/len(test_dataloader), recall_1

if __name__=="__main__":
    logging.basicConfig(format = "%(asctime)s - %(levelname)s - %(name)s - %(message)s",
                    datefmt = "%Y-%m-%d %H:%M:%S", level = logging.INFO)
    args = parse_args()
    ckpt_path = args.ckpt_path
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model, data_args = load_from_ckpt(args.ckpt_path)
    model.to(device)

    # load dataset from data_args
    logger.info("Data config:")
    neat_print_dict(data_args)
    logger.info("Data config ends")

    test_ds = PCapDataset.from_config(data_args,
                                      img_transform=model.clip_transform,
                                      split="test")
    test_dl = build_dataloader(test_ds, 
                               batch=args.batch_size,
                               num_workers=args.num_workers,
                               dist_training=False,
                               shuffle=False)
    loss, recall_1 = test_logic(model, test_dl, args.cand_source)
    recall_1_per = round(100*(recall_1/len(test_ds)), 2)
    logger.info("Eval done")
    logger.info(f"Correct: {recall_1}/{len(test_ds)}")
    logger.info(f"Recall@1: {recall_1_per}, loss: {loss}")
    