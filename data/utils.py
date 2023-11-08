import re, math, torch
import os.path as osp

from typing import Union, List, Dict
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

# Definion of types
VAL_TYPE = Union[str, List[str], List[List[str]]]

# copied from BLIP/data/utils.py
def pre_captions(caption: VAL_TYPE, 
                 max_words:int = 50) -> VAL_TYPE:
    
    if caption is None:
        return None
    elif isinstance(caption,str):
        return pre_caption_single(caption, max_words)

    assert isinstance(caption, list) and all(isinstance(c, str) for c in caption),\
            "Argument `caption` must be either str or List[str]"

    return [pre_caption_single(single, max_words) for single in caption]

def pre_caption_single(caption:str, max_words=50)->str:
    caption = re.sub(
        r"([.!\"()*#:;~?])",       
        ' ',
        caption.lower(),
    )
    caption = re.sub(
        r"\s{2,}",
        ' ',
        caption,
    )
    caption = caption.rstrip('\n') 
    caption = caption.strip(' ')

    #truncate caption
    caption_words = caption.split(' ')
    if math.isfinite(max_words) and len(caption_words) > max_words:
        caption = ' '.join(caption_words[:max_words])
    
    return caption

def collate_test_set(src: VAL_TYPE)->VAL_TYPE:
    '''
    merges additional_comments into comment
    '''

    assert "additional_comments" in src.keys()
    other_keys=[k for k in src.keys() if k not in ["comment","additional_comments"]]
    single_instance = isinstance(src["comment"],str)
    # other elements
    tgt={k:src[k] for k in other_keys}
    if single_instance:
        tgt["comment"]=[src["comment"], *src["additional_comments"]]
    else:
        tgt["comment"]=[items for items in zip(src["comment"], *src["additional_comments"])]
    
    del src["additional_comments"]

    return tgt

def img_hash_to_addr(src_dataset: Dict[str, VAL_TYPE], 
                     img_addr: str, 
                     img_name_fmt: str):
    '''
    src_dataset=[{
        "image_hash": paths,
        "comment": comment,
        "personality": persona
    }, ...]
    '''
    if "images" in src_dataset.keys():
        return src_dataset
    # 坑1: src_dataset is a dict, not dataset
    # 坑2: 使用dataloader加载的时候index是单个, 而不是一般想象中的batch

    src_dataset["images"]=osp.join(img_addr, 
                        img_name_fmt.format(src_dataset.pop("image_hash")))
    
    return src_dataset

    

def build_dataloader(src_dataset, 
                     batch: int, 
                     num_workers:int, 
                     dist_training:bool, sampler=None, shuffle=True):
    if dist_training:
        sampler=DistributedSampler(src_dataset)
    
    # collate function
    # TODO: add pre_caption
    def collate_fn(a: Dict[str, str]):
        max_keys = list(set([k for item in a for k in item.keys()]))
        max_keys.sort()
        
        a_union = {}
        # fetch ele from each record to build a union dict
        for k in max_keys:
            a_union[k] = [item[k] for item in a if k in item]
            # stack as tensor
            if all(isinstance(v, torch.Tensor) for v in a_union[k]):
                a_union[k] = torch.stack(a_union[k], dim=0)
            
        return a_union

    dataloader=DataLoader(src_dataset,
                    batch_size=batch, 
                    num_workers=num_workers,
                    pin_memory=True,
                    shuffle=(not dist_training) and shuffle,
                    sampler=sampler,
                    collate_fn=collate_fn)

    return dataloader