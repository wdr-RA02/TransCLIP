import json
import clip
import os.path as osp

from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
from typing import Dict, List, Union, Optional
from .utils import img_hash_to_addr, collate_test_set, pre_captions

class Personality_Captions(Dataset):
    def __init__(self, pcap_root: str, split: str,
                 img_transform,
                 max_len:int=30, 
                 split_dict:Dict[str,str] = None,
                 merge_test: bool=False,
                 **kwargs):
        super().__init__()

        self.pcap_root = pcap_root
        self.img_addr = osp.join(pcap_root, "yfcc_images")

        self.img_transform = img_transform
        self.text_transform = clip.tokenize

        if split_dict is None:
            split_dict={
                "train": "train.json",
                "val": "val.json",
                "test": "test.json"
            }

        if split == "eval":
            split = "val"
        if split not in split_dict:
            raise ValueError(f"split must be in [train, val, test], got {split}.")
        
        split_jsonfile = osp.join(pcap_root, "personality_captions", split_dict[split])
        print(f"Loading {split} split from {split_jsonfile}")
    
        # dataset: load from datasets.load_dataset
        with open(split_jsonfile) as fp:
            self.annotation=json.load(fp)
        
        if kwargs.get("debug", False):
            # use the first 1000 split for debugging the code
            print("Debug mode")
            self.annotation = self.annotation[:1000]

        # add persona list to self, for the model to load
        self.add_persona_lists()

        print(f"# Item: {len(self.annotation)}")
        print(f"# Persona: {len(self.persona_lists)}")

        # merge additional column into "comment"
        if merge_test and "additional_comments" in self.annotation[0].keys():
            self.annotation=list(map(collate_test_set, self.annotation))
        
        # ann_paths <=> config["img_path"]
        self.img_name_fmt="{}.jpg"

        # others
        self.max_len=max_len
    
    def add_persona_lists(self):
        '''
        obtain the list from $pcap_root/personality_captions/personalities.txt
        '''
        persona_txt = osp.join(self.pcap_root, "personality_captions", "personalities.txt")
        with open(persona_txt) as fp:
            self.persona_lists = [persona for persona in fp.read().strip().split("\n") \
                                     if "Traits" not in persona]

    def __getitem__(self, index):
        samples = self.annotation[index]
        # sample is Dict[str, whatever]
        
        item = img_hash_to_addr(samples, self.img_addr, self.img_name_fmt)
        item["comment"] = pre_captions(item["comment"], self.max_len)
        item["comment"] = self.text_transform(item.pop("comment")).squeeze(0)

        if "candidates" in item:
            item["candidates"] = pre_captions(item["candidates"], self.max_len)
            item["candidates"] = self.text_transform(item.pop("candidates"))
        
        # open image here to accel the process
        image = Image.open(item.pop("images"))
        item["images"] = self.img_transform(image).squeeze(0)

        '''
        out is dict of
        {
            "key" : [b1, b2, ...],
            ...
        }
        '''
        
        return item

    def __len__(self):
        return len(self.annotation)
    
    @classmethod
    def from_config(cls, data_cfg, img_transform, split: str="train"):
        # load a Dataset instance from data_cfg
        import os
        debug = (os.environ.get("DEBUG", 0)=="1")

        return cls(pcap_root = data_cfg["pcap_root"],
                   split = split,
                   img_transform = img_transform,
                   max_len = data_cfg["max_len"],
                   merge_test = data_cfg.get("merge_test", False),
                   split_dict = data_cfg["splits"],
                   debug = debug)
