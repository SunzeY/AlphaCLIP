import json
import os
import random
from tqdm import tqdm
from torch.utils.data import Dataset
from pycocotools.coco import COCO
from pycocotools import mask as maskUtils
from PIL import Image
import cv2
import random
from torchvision import transforms
from tqdm import tqdm

import pickle
import torch
import numpy as np
import copy
import sys
import shutil
from PIL import Image
from nltk.corpus import wordnet

PIXEL_MEAN = (0.48145466, 0.4578275, 0.40821073)
MASK_FILL = [int(255 * c) for c in PIXEL_MEAN]


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

class Imagenet_S(Dataset):
    def __init__(self, ann_file='data/imagenet_919.json', hi_res=False, all_one=False):
        self.anns = json.load(open(ann_file, 'r'))
        self.root_pth = 'data/'
        cats = []
        for ann in self.anns:
            if ann['category_word'] not in cats:
                cats.append(ann['category_word'])
            ann['cat_index'] = len(cats) - 1
        self.classes = []
        for cat_word in cats:
            synset = wordnet.synset_from_pos_and_offset('n', int(cat_word[1:]))
            synonyms = [x.name() for x in synset.lemmas()]
            self.classes.append(synonyms[0])
            
        self.choice = "center_crop"
        if hi_res:
            self.mask_transform = res_mask_transform
            self.clip_standard_transform = res_clip_standard_transform
        else:
            self.mask_transform = mask_transform
            self.clip_standard_transform = clip_standard_transform

        self.all_one = all_one

    def __len__(self):
        return len(self.anns)

    def __getitem__(self, index):
        ann = self.anns[index]
        image = cv2.imread(os.path.join(self.root_pth, ann['image_pth']))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = maskUtils.decode(ann['mask'])
        # image[mask==0] = MASK_FILL
        rgba = np.concatenate((image, np.expand_dims(mask, axis=-1)), axis=-1)
        h, w = rgba.shape[:2]
        
        if self.choice == "padding":
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
        # using box: bounding-box compute
        # bi_mask = mask == 1
        # h, w = bi_mask.shape[-2:]
        # in_height = np.max(bi_mask, axis=-1)
        # in_height_coords = np.max(bi_mask, axis=-1) * np.arange(h)
        # b_e = in_height_coords.max()
        # in_height_coords = in_height_coords + h * (~in_height)
        # t_e = in_height_coords.min()
        # in_width = np.max(bi_mask, axis=-2)
        # in_width_coords = np.max(bi_mask, axis=-2) * np.arange(w)
        # r_e = in_width_coords.max()
        # in_width_coords = in_width_coords + w * (~in_width)
        # l_e = in_width_coords.min()
        # box = np.zeros_like(mask)
        # box[t_e: b_e, l_e:r_e] = 1
        # mask = box
        if self.all_one:
            mask_torch = self.mask_transform(np.ones_like(mask) * 255)
        else: 
            mask_torch = self.mask_transform(mask * 255)
        return image_torch, mask_torch, ann['cat_index']

if __name__ == "__main__":
    data = Imagenet_S()
    for i in tqdm(range(data.__len__())):
        data.__getitem__(i)