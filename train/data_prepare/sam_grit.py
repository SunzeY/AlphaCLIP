from segment_anything import sam_model_registry, SamPredictor
import cv2
import numpy as np
import json
from PIL import Image, TarIO
import matplotlib.pyplot as plt
import torch
from typing import Any, Dict, Generator, ItemsView, List, Tuple
from itertools import groupby
from pycocotools import mask as mask_utils
from tqdm import tqdm
import os
import pickle
import tarfile
import argparse
import io

def coco_encode_rle(uncompressed_rle: Dict[str, Any]) -> Dict[str, Any]:
      # type: ignore
    h, w = uncompressed_rle["size"]
    rle = mask_utils.frPyObjects(uncompressed_rle, h, w)
    rle["counts"] = rle["counts"].decode("utf-8")  # Necessary to serialize with json
    return rle

def binary_mask_to_rle(binary_mask):
    rle = {'counts': [], 'size': list(binary_mask.shape)}
    counts = rle.get('counts')
    for i, (value, elements) in enumerate(groupby(binary_mask.ravel(order='F'))):
        if i == 0 and value == 1:
            counts.append(0)
        counts.append(len(list(elements)))
    return rle

parser = argparse.ArgumentParser()
parser.add_argument('--tar-pth', type=str, default="GRIT-1m/00001.tar")
args = parser.parse_args()
image_pths = []

sam_checkpoint = "sam_vit_h_4b8939.pth"
model_type = "vit_h"

device = "cuda"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

predictor = SamPredictor(sam)
result = dict()

with tarfile.open(args.tar_pth, 'r') as t:
    image_pths = [pth for pth in t.getnames() if pth[-4:] == '.jpg']

    for img in tqdm(image_pths):
        ann = json.load(t.extractfile(img.replace('.jpg', '.json')))
        tarinfo = t.getmember(img)
        image = t.extractfile(tarinfo)
        image = image.read()
        pil_img = Image.open(io.BytesIO(image)).convert("RGB")
        image_h = pil_img.height
        image_w = pil_img.width
        grounding_list = ann['ref_exps']
        try:
            predictor.set_image(np.array(pil_img))
        except:
            print(np.array(pil_img).shape)
            print(img)
            print(ann['id'])
        segs = []
        for i, (phrase_s, phrase_e, x1_norm, y1_norm, x2_norm, y2_norm, score) in enumerate(grounding_list):
            x1, y1, x2, y2 = int(x1_norm * image_w), int(y1_norm * image_h), int(x2_norm * image_w), int(y2_norm * image_h)
            input_box = np.array(([x1, y1, x2, y2]))
            masks, _, _ = predictor.predict(
                point_coords=None,
                point_labels=None,
                box=input_box[None, :],
                multimask_output=False,
            )
            rle = binary_mask_to_rle(np.asfortranarray(masks[0]))
            seg = coco_encode_rle(rle)
            segs.append(seg)
        result[ann['id']] = segs
    pickle.dump(result, \
                open(args.tar_pth.replace('GRIT-20m', 'GRIT_sam_masks_20m').replace('.tar', '.pkl'), 'wb'))
