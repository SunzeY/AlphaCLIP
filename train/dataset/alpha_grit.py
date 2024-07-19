import json
import os
import random
from tqdm import tqdm
from torch.utils.data import Dataset
from mask_image import ImageNet_Masked
from pycocotools.coco import COCO
from pycocotools import mask as maskUtils
from PIL import Image
import cv2
import random
from torchvision import transforms
from tqdm import tqdm
PIXEL_MEAN = (0.48145466, 0.4578275, 0.40821073)
MASK_FILL = [int(255 * c) for c in PIXEL_MEAN]
import pickle
import torch
import numpy as np
import copy
import sys
import shutil
from PIL import Image

def get_file(url):
    return #TODO: get file path from local directory

clip_standard_transform = transforms.Compose([
    transforms.ToTensor(), 
    transforms.Resize((224, 224), interpolation=Image.BICUBIC),
    transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
])

hi_clip_standard_transform = transforms.Compose([
    transforms.ToTensor(), 
    transforms.Resize((336, 336), interpolation=Image.BICUBIC),
    transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
])

res_clip_standard_transform = transforms.Compose([
    transforms.ToTensor(), 
    transforms.Resize((336, 336), interpolation=Image.BICUBIC),
    transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
])

mask_transform = transforms.Compose([
    transforms.ToTensor(), 
    transforms.Resize((224, 224)),
    transforms.Normalize(0.5, 0.26)
])

hi_mask_transform = transforms.Compose([
    transforms.ToTensor(), 
    transforms.Resize((336, 336)),
    transforms.Normalize(0.5, 0.26)
])

res_mask_transform = transforms.Compose([
    transforms.ToTensor(), 
    transforms.Resize((336, 336)),
    transforms.Normalize(0.5, 0.26)
])

def crop_center(img, croph, cropw):
    h, w = img.shape[:2]
    starth = h//2 - (croph//2)
    startw = w//2 - (cropw//2)    
    return img[starth:starth+croph, startw:startw+cropw, :]

class Alpha_GRIT(Dataset):
    def __init__(self, ids_file='grit_1m_ids.pkl', root_pth='grit-1m/', common_pair=0.0, hi_res=False, subnum=None):
        if subnum is not None:
            self.ids = pickle.load(open(ids_file, 'rb'))[:subnum]
        else:
            self.ids = pickle.load(open(ids_file, 'rb'))
        self.root_pth = root_pth
        self.with_common_pair_prop = common_pair
        if hi_res:
            self.mask_transform = res_mask_transform
            self.clip_standard_transform = res_clip_standard_transform
        else:
            self.mask_transform = mask_transform
            self.clip_standard_transform = clip_standard_transform
        
    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        id = self.ids[index]
        ann = json.loads(get_file(self.root_pth + str(id) + '.json'))
        image_data = get_file(self.root_pth + str(id) + '.jpg')
        img = np.frombuffer(image_data, dtype=np.uint8)
        img = cv2.imdecode(img, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ref_exps = ann['ref_exps']
        # random choose single ref with its corresponding masks
        choice = random.randint(0, len(ref_exps)-1)
        ref_exp = ref_exps[choice]
        text = ann['caption'][int(ref_exp[0]): int(ref_exp[1])]
        mask = maskUtils.decode(ann['seudo_masks'][choice])
        if mask.shape != img.shape[:2]:
            img = np.rot90(img)
        rgba = np.concatenate((img, np.expand_dims(mask, axis=-1)), axis=-1)
        h, w = rgba.shape[:2]
        choice = random.randint(0, 1)
        choice = 0
        if choice == 0:
            if max(h, w) == w:
                pad = (w - h) // 2
                l, r = pad, w - h - pad
                rgba = np.pad(rgba, ((l, r), (0, 0), (0, 0)), 'constant', constant_values=0)
            else:
                pad = (h - w) // 2
                l, r = pad, h - w - pad
                rgba = np.pad(rgba, ((0, 0), (l, r), (0, 0)), 'constant', constant_values=0)
        else:
            if min(h, w) == h:
                rgba = crop_center(rgba, h, h)
            else:
                rgba = crop_center(rgba, w, w)
        rgb = rgba[:, :, :-1]
        mask = rgba[:, :, -1]
        image_torch = self.clip_standard_transform(rgb)

        choice = random.random()
        if choice >= self.with_common_pair_prop:
            mask_torch = self.mask_transform(mask * 255)
            return image_torch, mask_torch, text 
        else: # half ori image
            mask_torch = self.mask_transform(np.ones_like(mask) * 255)
            return image_torch, mask_torch, ann['caption']