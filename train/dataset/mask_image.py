import json
import os
import random

from torch.utils.data import Dataset
from pycocotools.coco import COCO
from pycocotools import mask as maskUtils

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
from nltk.corpus import wordnet
from bg_aug import get_bkgd
import jax
import random

clip_standard_transform = transforms.Compose([
    transforms.ToTensor(), 
    transforms.Resize((224, 224), interpolation=Image.BICUBIC),
    transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
])
to_tensor = transforms.ToTensor()

normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))

mask_transform = transforms.Compose([
    transforms.ToTensor(), 
    transforms.Resize((224, 224)),
    transforms.Normalize(0.5, 0.26)
])

crop_aug = transforms.Compose([
    transforms.RandomCrop((224-32, 224-32)),
    transforms.Resize((224, 224)),
])

def text_filter(text):
    text = text.replace(' with a white background', '')
    text = text.replace(' with white background', '')
    text = text.replace(' next to a white background', '')
    text = text.replace(' over a white background', '')
    text = text.replace(' is cut out of a white background', '')
    text = text.replace(' across a white background', '')
    text = text.replace(' on a white background', '')
    text = text.replace(' sticking out of a white background', '')
    text = text.replace(' in the middle of a white background', '')
    text = text.replace(' on white background', '')
    text = text.replace(' in a white background', '')
    text = text.replace(' and a white background', '')
    text = text.replace(' and white background', '')
    text = text.replace(' in front of a white background', '')
    text = text.replace(' on top of a white background', '')
    text = text.replace(' against a white background', '')
    text = text.replace('a white background with ', '')
    text = text.replace(' and has a white background', '')
    text = text.replace('white background', 'background')
    text = text + '.'
    return text

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
    cropped_image[np.nonzero(cropped_mask == 0)] = masked_color
    return cropped_image, cropped_mask

class ImageNet_Masked(Dataset):
    def __init__(self, ann_file="M_ImageNet_top_460k.json",  masked_color=[255, 255, 255]):
        self.masked_color = masked_color
        self.anns_list = json.load(open(ann_file, 'r'))
        random.shuffle(self.anns_list)
        self.crop_scale = 1.5
        self.transform = clip_standard_transform
        self.res = 224
        self.blur = 10.0

    def __len__(self):
        return len(self.anns_list)

    def __getitem__(self, index):
        cv2.ocl.setUseOpenCL(False)
        cv2.setNumThreads(0)
        ann = self.anns_list[index]
        # TODO: change list to dict key.
        img_pth = ann[2]
        # img_pth = img_pth.replace('imagenet-21k/images', 'imagenet-21k-demo/*')
        mask = ann[3]
        bbox = ann[4]
        text = ann[6]
        image = cv2.imread(img_pth)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        bbox_xywh = np.copy(np.array(bbox))
        binary_mask = maskUtils.decode(mask)
        cat_word = img_pth.split("/")[3]
        synset = wordnet.synset_from_pos_and_offset('n', int(cat_word[1:]))
        synonyms = [x.name() for x in synset.lemmas()]
        text = text.replace(".", f", probably {synonyms[0]}").replace(" ", "_").replace("/", "_").replace("\\", "_")
        image[np.nonzero(binary_mask == 1)] = (0.5 * image[np.nonzero(binary_mask == 1)] + 0.5 * np.array([0, 255, 0])).astype(np.uint8) 
        os.makedirs(os.path.split(img_pth.replace("imagenet-21k/images", "visual_train_c"))[0], exist_ok=True)
        Image.fromarray(image).save(os.path.split(img_pth.replace("imagenet-21k/images", "visual_train_c"))[0] + f"/{text}_" + os.path.split(img_pth.replace("imagenet-21k/images", "visual_train_c"))[1])

if __name__ == "__main__":
    data = ImageNet_Masked()
    for i in tqdm(range(data.__len__())):
        data.__getitem__(i)
