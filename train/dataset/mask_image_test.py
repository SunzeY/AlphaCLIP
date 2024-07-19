import json
import os
import random

from torch.utils.data import Dataset
from pycocotools.coco import COCO
from pycocotools import mask as maskUtils
from lvis import LVIS
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None
from tqdm import tqdm
from torchvision import transforms
from tqdm import tqdm
import pickle
import cv2
import torch
import numpy as np
import copy
from transformers import AutoProcessor
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

def crop(image: np.array, bbox_xywh: np.array, bi_mask: np.array, scale=1.5):
    tl_x = int(bbox_xywh[0])
    tl_y = int(bbox_xywh[1])
    w = int(bbox_xywh[2]) if int(bbox_xywh[2]) > 0 else 1
    h = int(bbox_xywh[3]) if int(bbox_xywh[3]) > 0 else 1
    image_h, image_w = image.shape[:2]

    # shape maintained
    r = max(h, w)
    tl_x -= (r - w) / 2
    tl_y -= (r - h) / 2
    half_scale = (scale - 1.0) / 2
    w_l = int(tl_x - half_scale * r) if (tl_x - half_scale * r) > 0 else 0
    w_r = int(tl_x + (1+half_scale) * r) if (tl_x + (1+half_scale) * r) < image_w else image_w - 1
    h_t = int(tl_y - half_scale * r) if (tl_y - half_scale * r) > 0 else 0
    h_b = int(tl_y + (1+half_scale) * r) if (tl_y + (1+half_scale) * r) < image_h else image_h - 1

    return image[h_t: h_b, w_l: w_r, :], bi_mask[h_t: h_b, w_l: w_r]

def masked_crop(image: np.array, bbox_xywh: np.array, bi_mask: np.array, crop_scale=1.0, masked_color=[255, 255, 255]):
    # padding to make_sure bboxshape maintained
    image = np.pad(image, ((600, 600), (600, 600), (0, 0)), 'constant', constant_values=255)
    bi_mask = np.pad(bi_mask, ((600, 600), (600, 600)), "constant", constant_values=0)
    bbox_xywh[:2] += 600
    cropped_image, cropped_mask = crop(image, bbox_xywh, bi_mask, crop_scale)   
    # cropped_image[np.nonzero(cropped_mask == 0)] = MASK_FILL
    return cropped_image, cropped_mask

class COCO_Masked_Test(Dataset):
    def __init__(self, ann_file="data/coco/annotations/instances_val2017.json",  masked_color=[255, 255, 255], root_directory="data/coco/val2017", hi_res=False):
        self.masked_color = masked_color
        self.coco = COCO(annotation_file=ann_file)
        self.image_directory = root_directory
        self.crop_scale = 1.5
        self.anns_list = list(self.coco.anns.keys())
        self.index2id = [x['id'] for x in self.coco.cats.values()]
        self.id2index = dict()
        for i, item in enumerate(self.index2id):
            self.id2index[item] = i
        self.class_num = 80
        self.classes = [x['name'] for x in self.coco.cats.values()]
        
        if hi_res:
            self.mask_transform = hi_mask_transform
            self.clip_standard_transform = hi_clip_standard_transform
        else:
            self.mask_transform = mask_transform
            self.clip_standard_transform = clip_standard_transform
        
    def __len__(self):
        return len(self.anns_list)

    def __getitem__(self, index):
        ann_id = self.anns_list[index]
        ann = self.coco.anns[ann_id]
        img_id = self.coco.anns[ann_id]['image_id']
        image = np.array(Image.open(os.path.join(self.image_directory, self.coco.imgs[img_id]['file_name'])).convert('RGB'))
        bbox_xywh = np.copy(np.array(ann['bbox']))
        binary_mask = self.coco.annToMask(ann)
        cropped_image, cropped_mask =  masked_crop(image, bbox_xywh, binary_mask, crop_scale=self.crop_scale, masked_color=self.masked_color)
        image = self.clip_standard_transform(cropped_image)
        mask_torch = self.mask_transform(cropped_mask * 255)
        return image, mask_torch, self.id2index[ann['category_id']]

class LVIS_Masked_Test(Dataset):
    def __init__(self, ann_file="data/lvis/annotations/lvis_v1_val.json",  masked_color=[255, 255, 255], hi_res=False):
        self.masked_color = masked_color
        self.lvis = LVIS(ann_file)
        self.crop_scale = 1.5
        self.anns_list = list(self.lvis.anns.keys())
        self.index2id = [x['id'] for x in self.lvis.cats.values()]
        self.id2index = dict()
        for i, item in enumerate(self.index2id):
            self.id2index[item] = i
        self.class_num = 1203
        self.classes = [x['name'] for x in self.lvis.cats.values()]
        
        if hi_res:
            self.mask_transform = hi_mask_transform
            self.clip_standard_transform = hi_clip_standard_transform
        else:
            self.mask_transform = mask_transform
            self.clip_standard_transform = clip_standard_transform

    def __len__(self):
        return len(self.anns_list)

    def __getitem__(self, index):
        ann_id = self.anns_list[index]
        ann = self.lvis.anns[ann_id]
        img_id = self.lvis.anns[ann_id]['image_id']
        image = np.array(Image.open(self.lvis.imgs[img_id]['coco_url'].replace('http://images.cocodataset.org', 'data/coco')).convert('RGB'))
        binary_mask = self.lvis.ann_to_mask(ann)
        rgba = np.concatenate((image, np.expand_dims(binary_mask, axis=-1)), axis=-1)
        h, w = rgba.shape[:2]
        if max(h, w) == w:
            pad = (w - h) // 2
            l, r = pad, w - h - pad
            rgba = np.pad(rgba, ((l, r), (0, 0), (0, 0)), 'constant', constant_values=0)
        else:
            pad = (h - w) // 2
            l, r = pad, h - w - pad
            rgba = np.pad(rgba, ((0, 0), (l, r), (0, 0)), 'constant', constant_values=0)
        rgb = rgba[:, :, :-1]
        mask = rgba[:, :, -1]
        image = self.clip_standard_transform(rgb)
        mask_torch = self.mask_transform(mask * 255)
        return image, mask_torch, self.id2index[ann['category_id']], 

if __name__ == "__main__":
    data = LVIS_Masked_Test()
    for i in tqdm(range(data.__len__())):
        data.__getitem__(i)