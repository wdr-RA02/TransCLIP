import clip
import torch
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image
from typing import List, Optional, Union

def clip_loss(score):
    logprob = F.log_softmax(score, dim=1)
    logprob_tr = F.log_softmax(score.t(), dim=1)
    targets = torch.arange(0, logprob.shape[1], dtype=torch.long, 
                           device=logprob.device)
    
    loss = (F.nll_loss(logprob, targets) + F.nll_loss(logprob_tr, targets))/2.

    return loss

class TransCLIPModel(nn.Module):
    def __init__(self, 
                 persona_list: List[str],
                 args: dict,
                 **kwargs) -> None:
        super().__init__()

        # model misc args
        self.args = args
        # init CLIP text and img encoder
        clip_model = args["clip_type"]
        print(f"Loading TransCLIP with CLIP Model: {clip_model}")
        
        self.clip_model, self.clip_transform = clip.load(clip_model)
        self.clip_model.to(dtype=torch.float32)

        self.img_dim = self.clip_model.visual.proj.shape[1]
        self.text_dim = self.clip_model.text_projection.shape[1]

        # persona_list
        self.persona_list = persona_list
        self.n_persona = len(self.persona_list)+1
        # the last is reserved for unk personality
        self.persona_list_to_dict()

        # feed forward for img feat
        self.hidden_dim = self.args["hidden_dim"]
        n_layers_img = self.args["num_layers_img"]
        n_layers_txt = self.args["num_layers_txt"]

        # build feed forward layers like TransResNet do
        self.build_img_feedfwd(n_layers_img)
        self.build_persona_ln()
        self.build_text_feedfwd(n_layers_txt)

        self.device = self.clip_model.visual.proj.device

    @property
    def dtype(self):
        return self.clip_model.dtype

    def build_img_feedfwd(self, n_layers_img:int):
        if not 0<=self.args["dropout"]<1.0:
            raise ValueError("Drop rate must be in the range of [0.0, 1.0)")
        # hidden_clip -> hidden_dim first
        ff_layers = [
            # nn.BatchNorm1d(self.img_dim),
            nn.LayerNorm(self.img_dim),
            nn.Dropout(p=self.args["dropout"]),
            nn.Linear(self.img_dim, self.hidden_dim)
        ]
        
        ff_layers += [
            nn.ReLU(),
            nn.Dropout(p=self.args["dropout"]),
            nn.Linear(self.hidden_dim, self.hidden_dim)
        ] * (n_layers_img-1)

        self.img_feedfwd = nn.Sequential(*ff_layers).to(dtype=self.clip_model.dtype)

    def build_persona_ln(self):
        persona_ln = [
            # nn.BatchNorm1d(self.n_persona),
            nn.LayerNorm(self.n_persona),
            nn.Dropout(p=self.args["dropout"]),
            nn.Linear(self.n_persona, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim)
        ]

        self.personality_enc = nn.Sequential(*persona_ln).to(dtype=self.clip_model.dtype)
    
    def build_text_feedfwd(self, n_layers_txt:int):
        # hidden_clip -> hiddem_dim
        # MLP-Like structure
        ff_layers=[
            nn.LayerNorm(self.text_dim),
            nn.Dropout(p=self.args["dropout"]),
            nn.Linear(self.text_dim, self.hidden_dim),
        ]

        ff_layers+=[
            nn.ReLU(),
            nn.Dropout(p=self.args["dropout"]),
            nn.Linear(self.hidden_dim, self.hidden_dim)
        ] * n_layers_txt

        self.text_feedfwd = nn.Sequential(*ff_layers).to(dtype=self.clip_model.dtype)

    def persona_list_to_dict(self):
        self.persona_dict = {p:i for i,p in enumerate(self.persona_list)}

    def encode_imgs(self, imgs:List[str], transform = None) -> torch.Tensor:
        if isinstance(imgs, torch.Tensor):
            return imgs.to(device=self.device, non_blocking=True)
        elif not isinstance(imgs, list):
            raise TypeError(f"Arg img must be List[str] or Tensor, got {type(imgs)}. ")
        
        if transform is None:
            transform = self.clip_transform
        
        img = [transform(Image.open(im).convert("RGB")) for im in imgs]
        img_tensor = torch.stack(img, dim=0).to(device=self.device, non_blocking=True)
        
        return img_tensor

    
    def encode_texts(self, txts: List[str]) -> torch.LongTensor:
        texts = clip.tokenize(txts, truncate=True).to(self.device, non_blocking=True)

        return texts

    def forward_persona(self, personas:List[str]) -> torch.Tensor:
        '''
        convert personality lists to a one-hot matrix of [b, n_persona]

        and pass it to the personality encoder

        args:
        - personas: List containing persona strs

        returns:
        - persona_feat: encoded persona feature
        - persona_onehots: the one-hot vecs of the personas in this batch
        '''

        batch = len(personas)
        persona_onehots = torch.zeros(batch, self.n_persona, 
                                      dtype=self.clip_model.dtype).to(self.device, non_blocking=True)
        for i, item in enumerate(personas):
            p_index = self.persona_dict.get(item, self.n_persona-1)
            # set the corr persona to 1
            persona_onehots[i, p_index] = 1
            # print(p_index)
        
        # send the one-hot vecs to persona encoder
        persona_feat = self.personality_enc(persona_onehots)

        return persona_feat, persona_onehots
    
    def forward(self,
                pixel_values: torch.Tensor,
                captions: torch.LongTensor,
                personas: Optional[List[str]] = None,
                ):
        '''
        args:
        - pixel_values: list with torch.Tensor of shape [3, p, p]
        - captions: torch.Tensor of shape [b, max_len]
        - personas: list with b str
        '''
        # img_feat = self.img_feat(pixel_values)
        # txt_feat = self.txt_encoder(captions, captions_attn)
        # persona_feat = ...self.personality_enc(personas)
        # out = img_feat + txt_feat

        persona_contain = False
        if personas is not None:
            assert isinstance(personas, list) and len(personas)>0 and \
                all([isinstance(p, str) for p in personas]), "persona list check failed"
            assert len(personas)==pixel_values.shape[0], "persona and img must be of equal len"
            persona_contain = True

        img_feat = self.clip_model.encode_image(pixel_values)
        img_feat = self.img_feedfwd(img_feat)
        img_feat = F.normalize(img_feat, p=2, dim=1)

        txt_feat = self.clip_model.encode_text(captions)
        txt_feat = self.text_feedfwd(txt_feat)
        txt_feat = F.normalize(txt_feat, p=2, dim=1)

        # ensure personas is a List[str]
        persona_feat = None
        if persona_contain:
            '''persona feature here'''
            persona_feat, _ = self.forward_persona(personas)
            persona_feat = F.normalize(persona_feat, p=2, dim=1)

        return {
            "img_feature": img_feat, 
            "text_feature": txt_feat, 
            "personality_feature": persona_feat
        }
    
    def forward_batch(self, imgs, captions, personas, training:bool = True):
        '''
        during training, input imgs, captions and personas 
        and return the nll loss that maximizes the retrieval score
        between the paired imgs

        args: 
        - imgs: list of image addrs
        - captions: list of captions
        - personas: list of personas

        returns:
        - loss: the nll loss of predicted retrieval score wrt to real
        - num_correct: correct predictions in this batch
        '''
        if training:
            self.train()
        
        pixels = self.encode_imgs(imgs)
        caption_tokens = self.encode_texts(captions)

        features = self.forward(pixel_values=pixels,
                                captions=caption_tokens,
                                personas=personas)
        img_per = F.normalize((features["img_feature"] + features["personality_feature"]), p=2, dim=-1)
        score =  img_per @ features["text_feature"].t() 

        # obtain logprob of score
        loss = clip_loss(score * self.clip_model.logit_scale.exp())
        logprob = F.log_softmax(score, dim=1)
        targets = torch.arange(0, logprob.shape[1], dtype=torch.long, device=self.device)
        num_correct = int((torch.max(logprob, dim=1)[1]==targets).cpu().sum())

        return loss, num_correct

    @torch.no_grad()
    def eval_batch(self, imgs, personas, gt_captions, candidates):
        '''
        args:
        - gt_captions: ds["comment"]
        - candidiates: ds["candidiates"], with gt_caption inside
        
        returns:
        - loss: test loss, mean reduced
        - corr: # of correct predictions for this batch
        '''
        self.eval()
        # get ranking of gold caption in candidates
        assert len(gt_captions) == len(candidates)

        gt_rank = [item.index(gt) for item, gt in zip(candidates, gt_captions)]

        # forward prop
        pixels = self.encode_imgs(imgs)
        caption_tokens = [self.encode_texts(cand_str) for cand_str in candidates]
        # caption_tokens = [100*L*D]
        # stack the caption_tokens
        batch = len(gt_captions)
        txt_dim = caption_tokens[0].shape[-1]
        caption_tokens = torch.stack(caption_tokens, dim=0).reshape(-1, txt_dim)

        # perform forward prop
        features = self.forward(pixel_values=pixels,
                                captions=caption_tokens,
                                personas=personas)
        
        cand_feature = features["text_feature"].reshape(batch, -1, self.hidden_dim)
        img_per = F.normalize((features["img_feature"] + features["personality_feature"]), p=2, dim=-1)
        # cand_feature = [100*b, self.txt_dim] -> [b, 100, self.txt_dim]
        score = torch.bmm(
            img_per.unsqueeze(1),
            cand_feature.transpose(-1,-2),
        ).squeeze(1)

        # score = [b, (1,) 100]
        # fetch top-1 from score
        topk_indexes = score.topk(k=1, dim=-1)[1].cpu().squeeze(1)
        target_indexes = torch.tensor(gt_rank).to(dtype=topk_indexes.dtype)
        corr = int((topk_indexes==target_indexes).sum())

        # test loss
        # obtain logprob of score
        logprob = F.log_softmax(score, dim=1).cpu()
        targets = torch.tensor(gt_rank, dtype=torch.long)
        loss = F.nll_loss(logprob, targets)

        return loss, corr