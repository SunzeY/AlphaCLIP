from typing import List, Dict, Union, Tuple

from PIL import Image, ImageDraw, ImageFilter, ImageOps, ImageEnhance
import spacy
import hashlib
import os

import torch
import torchvision
import torchvision.transforms as transforms
import clip
from transformers import BertTokenizer, RobertaTokenizerFast
import ruamel.yaml as yaml
import copy

from interpreter import Box

import pycocotools.mask as mask_utils
import alpha_clip
from segment_anything import sam_model_registry, SamPredictor
import numpy as np
import cv2
import matplotlib.pyplot as plt

import pickle

class Executor:
    def __init__(self, device: str = "cpu", box_representation_method: str = "crop", method_aggregator: str = "max", enlarge_boxes: int = 0, expand_position_embedding: bool = False, square_size: bool = False, blur_std_dev: int = 100, cache_path: str = None, input_file: str = None) -> None:
        IMPLEMENTED_METHODS = ["blur", "full", "gray"]
        if any(m not in IMPLEMENTED_METHODS for m in box_representation_method.split(",")):
            raise NotImplementedError
        IMPLEMENTED_AGGREGATORS = ["max", "sum"]
        if method_aggregator not in IMPLEMENTED_AGGREGATORS:
            raise NotImplementedError
        self.box_representation_method = box_representation_method
        self.method_aggregator = method_aggregator
        self.enlarge_boxes = enlarge_boxes
        self.device = device
        self.expand_position_embedding = expand_position_embedding
        self.square_size = square_size
        self.blur_std_dev = blur_std_dev
        self.cache_path = cache_path

    def preprocess_image(self, image: Image) -> List[torch.Tensor]:
        return [preprocess(image) for preprocess in self.preprocesses]
    
    def preprocess_mask(self, mask: Image) -> List[torch.Tensor]:
        preprocess = self.preprocesses[0]
        return preprocess.transforms[1](preprocess.transforms[0](mask)) 

    def preprocess_text(self, text: str) -> torch.Tensor:
        raise NotImplementedError

    def call_model(self, model: torch.nn.Module, images: torch.Tensor, text: Union[torch.Tensor, Dict[str, torch.Tensor]]) -> torch.Tensor:
        raise NotImplementedError

    def tensorize_inputs(self, caption: str, image: Image, boxes: List[Box], image_name: str = None, image_pth: str = None) -> Tuple[List[torch.Tensor], torch.Tensor]:
        images = []
        for preprocess in self.preprocesses:
            images.append([])
        
        if 'aclip' in self.clip_type:
            self.all_masks = []
            read_save = False
            if self.mask_path is not None: # load mask if cached
                file_name = image_pth.split('/')[-1].split('.')[0]+'.pkl'
                if os.path.exists(os.path.join(self.mask_path, file_name)):
                    all_rles = pickle.load(open(os.path.join(self.mask_path, file_name),'rb'))
                    for rle in all_rles:
                        mask = np.array(mask_utils.decode(rle), dtype=bool)
                        self.all_masks.append(mask)
                    read_save = True 
            if not read_save:
                # use SAM to generate masks
                self.predictor.set_image(np.array(image.convert('RGB')))
                all_rles = []
                for i in range(len(boxes)):
                    box = [
                        max(boxes[i].left-self.enlarge_boxes, 0),
                        max(boxes[i].top-self.enlarge_boxes, 0),
                        min(boxes[i].right+self.enlarge_boxes, image.width),
                        min(boxes[i].bottom+self.enlarge_boxes, image.height)
                    ] # box prompt
                    input_box = np.array(box)
                    masks, _, _ = self.predictor.predict(
                        point_coords=None,
                        point_labels=None,
                        box=input_box[None, :],
                        multimask_output=False,
                    )
                    self.all_masks.append(masks[0])
                    rle = mask_utils.encode(np.array(masks[0][:, :, None], order='F', dtype="uint8"))[0]
                    rle["counts"] = rle["counts"].decode("utf-8")
                    all_rles.append(rle)
                if self.mask_path is not None: # save mask
                    os.makedirs(self.mask_path, exist_ok=True)
                    pickle.dump(all_rles, open(os.path.join(self.mask_path, file_name),'wb'))

        if self.cache_path is None or any([not os.path.exists(os.path.join(self.cache_path, "refcoco_val", model_name, "image", image_name, method_name+".pt")) for model_name in self.model_names for method_name in self.box_representation_method.split(',')]): 
            if "full" in self.box_representation_method: # original full image with alpha-map
                for i in range(len(boxes)):
                    image_i = image.copy()
                    preprocessed_images = self.preprocess_image(image_i)
                    for j, img in enumerate(preprocessed_images):
                        images[j].append(img.to(self.device))
            if "blur" in self.box_representation_method:
                for i in range(len(boxes)):
                    image_i = image.copy()

                    mask = Image.new('L', image_i.size, 0)
                    draw = ImageDraw.Draw(mask)
                    box = (
                        max(boxes[i].left-self.enlarge_boxes, 0),
                        max(boxes[i].top-self.enlarge_boxes, 0),
                        min(boxes[i].right+self.enlarge_boxes, image_i.width),
                        min(boxes[i].bottom+self.enlarge_boxes, image_i.height)
                    )
                    if 'aclip' in self.clip_type:
                        width, height = image.size
                        for y in range(height):
                            for x in range(width):
                                if self.all_masks[i][y][x] == 1:
                                    draw.point((x, y), fill=255)
                    else:
                        draw.rectangle([box[:2], box[2:]], fill=255)
                    blurred = image_i.filter(ImageFilter.GaussianBlur(self.blur_std_dev))
                    blurred.paste(image_i, mask=mask)
                    preprocessed_images = self.preprocess_image(blurred)

                    for j, img in enumerate(preprocessed_images):
                        images[j].append(img.to(self.device))
            if "gray" in self.box_representation_method:
                for i in range(len(boxes)):
                    image_i = image.copy()
                    mask_i = self.all_masks[i]
                    width, height = image.size

                    pixels = image_i.load()
                    for y in range(height):
                        for x in range(width):
                            if mask_i[y][x] == 0:
                                pixel_value = pixels[x, y]
                                gray_value = int(0.2989 * pixel_value[0] + 0.5870 * pixel_value[1] + 0.1140 * pixel_value[2])
                                pixels[x, y] = (gray_value, gray_value, gray_value)
                    preprocessed_images = self.preprocess_image(image_i)
                    for j, img in enumerate(preprocessed_images):
                        images[j].append(img.to(self.device))

            imgs = [torch.stack(image_list) for image_list in images]
        else:
            imgs = [[] for _ in self.models]
        text_tensor = self.preprocess_text(caption.lower()).to(self.device)
        return imgs, text_tensor

    @torch.no_grad()
    def __call__(self, caption: str, image: Image, boxes: List[Box], image_name: str = None, image_pth=None) -> torch.Tensor:
        images, text_tensor = self.tensorize_inputs(caption, image, boxes, image_name, image_pth) 
        all_logits_per_image = []
        all_logits_per_text = []
        box_representation_methods = self.box_representation_method.split(',')
        caption_hash = hashlib.md5(caption.encode('utf-8')).hexdigest() 
        for model, images_t, model_name in zip(self.models, images, self.model_names):
            self.image_feat_path = ""
            if self.cache_path is not None:
                text_cache_path = os.path.join(self.cache_path, "refcoco_val", model_name, "text"+("_shade" if self.box_representation_method == "shade" else ""))
                image_feat_path = os.path.join(self.cache_path, "refcoco_val", model_name, "image", image_name)
                self.image_feat_path = image_feat_path
            image_features = None
            text_features = None
            if self.cache_path is not None and os.path.exists(os.path.join(self.cache_path, "refcoco_val", model_name)):
                if os.path.exists(os.path.join(text_cache_path, caption_hash+".pt")):
                    text_features = torch.load(os.path.join(text_cache_path, caption_hash+".pt"), map_location=self.device)
                if os.path.exists(image_feat_path): 
                    if all([os.path.exists(os.path.join(image_feat_path, method_name+".pt")) for method_name in box_representation_methods]):
                        image_features = []
                        for method_name in box_representation_methods:
                            features = torch.load(os.path.join(image_feat_path, method_name+".pt"), map_location=self.device)
                            image_features.append(torch.stack([
                                features[(box.x, box.y, box.w, box.h)]
                                for box in boxes
                            ]))
                        image_features = torch.stack(image_features)
                        image_features = image_features.view(-1, image_features.shape[-1])
            logits_per_image, logits_per_text, image_features, text_features = self.call_model(model, images_t, text_tensor, image_features=image_features, text_features=text_features, boxes=boxes, image_pth=image_pth)
            all_logits_per_image.append(logits_per_image) 
            all_logits_per_text.append(logits_per_text) 
            if self.cache_path is not None and image_name is not None and image_features is not None:
                image_features = image_features.view(len(box_representation_methods), len(boxes), image_features.shape[-1]) 
                if not os.path.exists(image_feat_path):
                    os.makedirs(image_feat_path)
                for i in range(image_features.shape[0]):
                    method_name = box_representation_methods[i]
                    if not os.path.exists(os.path.join(image_feat_path, method_name+".pt")):
                        image_features_dict = {(box.x, box.y, box.w, box.h): image_features[i,j,:].cpu() for j, box in enumerate(boxes)} 
                        torch.save(image_features_dict, os.path.join(image_feat_path, method_name+".pt")) 
            if self.cache_path is not None and not os.path.exists(os.path.join(text_cache_path, caption_hash+".pt")) and text_features is not None:
                assert text_features.shape[0] == 1
                if not os.path.exists(text_cache_path):
                    os.makedirs(text_cache_path)
                torch.save(text_features.cpu(), os.path.join(text_cache_path, caption_hash+".pt"))

        all_logits_per_image = torch.stack(all_logits_per_image).sum(0)
        all_logits_per_text = torch.stack(all_logits_per_text).sum(0)
        if self.method_aggregator == "max":
            all_logits_per_text = all_logits_per_text.view(-1, len(boxes)).max(dim=0, keepdim=True)[0]
        elif self.method_aggregator == "sum":
            all_logits_per_text = all_logits_per_text.view(-1, len(boxes)).sum(dim=0, keepdim=True) 
        return all_logits_per_text.view(-1) 

class ClipExecutor(Executor):
    def __init__(self, clip_model: str = "ViT-B/32", device: str = "cpu", box_representation_method: str = "crop", method_aggregator: str = "max", enlarge_boxes: int = 0, expand_position_embedding: bool = False, square_size: bool = False, blur_std_dev: int = 100, cache_path: str = None, input_file: str = None, clip_type: str=None) -> None:
        super().__init__(device, box_representation_method, method_aggregator, enlarge_boxes, expand_position_embedding, square_size, blur_std_dev, cache_path)
        self.clip_models = clip_model.split(",")
        self.model_names = [model_name.replace("/", "_") for model_name in self.clip_models]
        self.models = []
        self.preprocesses = []
        self.data_name = input_file.split('/')[-1].split('.')[0]
        self.mask_path = None
        self.clip_type = clip_type
        if self.cache_path is not None:
            self.mask_path = os.path.join(self.cache_path, "refcoco_val", 'det_masks')
        sam_checkpoint = "./ckpt/sam_vit_h_4b8939.pth"
        model_type = "vit_h"
        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        sam.to(device=device)
        self.predictor = SamPredictor(sam)
        for model_name in self.clip_models:
            if 'aclip' in self.clip_type:#using alpha-clip
                self.mask_transform = transforms.Compose([
                    transforms.ToTensor(), 
                    transforms.Resize((224, 224)),
                    transforms.Normalize(0.5, 0.26)
                ]) 
                if model_name == 'ViT-B/16':
                    model, preprocess = alpha_clip.load("ViT-B/16", alpha_vision_ckpt_pth="./ckpt/grit1m/clip_b16_grit+mim_fultune_4xe.pth", device=device)
                elif model_name == 'ViT-L/14':
                    model, preprocess = alpha_clip.load("ViT-L/14", alpha_vision_ckpt_pth="./ckpt/grit1m/clip_l14_grit+mim_fultune_6xe.pth", device=device) 
               
            else: model, preprocess = clip.load(model_name, device=device, jit=False)
            self.models.append(model)
            if self.square_size:
                print("Square size!")
                preprocess.transforms[0] = transforms.Resize((model.visual.input_resolution, model.visual.input_resolution), interpolation=transforms.InterpolationMode.BICUBIC)
            self.preprocesses.append(preprocess)
        self.models = torch.nn.ModuleList(self.models)

    def preprocess_text(self, text: str) -> torch.Tensor:
        if "aclip" in self.box_representation_method:
            return alpha_clip.tokenize([text.lower()])
        if "shade" in self.box_representation_method:
            return clip.tokenize([text.lower()+" is in red color."])
        return clip.tokenize(["a photo of "+text.lower()])

    def call_model(self, model: torch.nn.Module, images: torch.Tensor, text: torch.Tensor, image_features: torch.Tensor = None, text_features: torch.Tensor = None, boxes=None, image_pth=None) -> torch.Tensor:
        if image_features is None:
            print('computing image features')
            if 'aclip' not in self.clip_type:
                image_features = model.encode_image(images)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            else:
                image_features = []
                if 'full' in self.box_representation_method:
                    aclip_images = images[:len(boxes)]
                    alphas = []

                    if os.path.exists(os.path.join(self.image_feat_path, 'full.pt')):
                        features = torch.load(os.path.join(self.image_feat_path, 'full.pt'), map_location=self.device)
                        aclip_image_features = torch.stack([
                            features[(box.x, box.y, box.w, box.h)]
                            for box in boxes
                        ])
                    else:
                        for i in range(len(self.all_masks)):
                            binary_mask = self.all_masks[i] 
                            alpha = self.mask_transform((binary_mask * 255).astype(np.uint8)) 
                            alpha = alpha.half().cuda().unsqueeze(dim=0)
                            alphas.append(alpha)
                        
                        alphas = torch.cat(alphas, dim=0)
                        aclip_images = aclip_images.half()
                        aclip_image_features = model.visual(aclip_images, alphas) # using alpha channels
                    images = images[len(boxes):]
                    image_features.append(aclip_image_features)

                if 'blur' in self.box_representation_method:
                    if os.path.exists(os.path.join(self.image_feat_path, 'blur.pt')):
                        features = torch.load(os.path.join(self.image_feat_path, 'blur.pt'), map_location=self.device)
                        ablur_images_features = torch.stack([
                            features[(box.x, box.y, box.w, box.h)]
                            for box in boxes
                        ])
                    else:
                        ablur_images = images[:len(boxes)]
                        alphas = []
                        for i in range(len(self.all_masks)):
                            binary_mask = self.all_masks[i]
                            alpha = self.mask_transform((binary_mask * 255).astype(np.uint8))
                            alpha = alpha.half().cuda().unsqueeze(dim=0)
                            alphas.append(alpha)
                        alphas = torch.cat(alphas, dim=0)
                        ablur_images = ablur_images.half()
                        ablur_images_features = model.visual(ablur_images, alphas)
                    images = images[len(boxes):]
                    image_features.append(ablur_images_features)

                if 'gray' in self.box_representation_method:
                    if os.path.exists(os.path.join(self.image_feat_path, 'gray.pt')):
                        features = torch.load(os.path.join(self.image_feat_path, 'gray.pt'), map_location=self.device)
                        gray_images_features = torch.stack([
                            features[(box.x, box.y, box.w, box.h)]
                            for box in boxes
                        ])
                    else:
                        gray_images = images[:len(boxes)]
                        alphas = []
                        for i in range(len(self.all_masks)):
                            binary_mask = self.all_masks[i]
                            alpha = self.mask_transform((binary_mask * 255).astype(np.uint8))
                            alpha = alpha.half().cuda().unsqueeze(dim=0)
                            alphas.append(alpha)
                        alphas = torch.cat(alphas, dim=0)
                        gray_images = gray_images.half()
                        gray_images_features = model.visual(gray_images, alphas)
                    images = images[len(boxes):]
                    image_features.append(gray_images_features)


                image_features = torch.cat(image_features, dim=0)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                
        if text_features is None:
            print('computing text features')
            text_features = model.encode_text(text)
            # normalized features
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # cosine similarity as logits
        logit_scale = model.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()
        return logits_per_image, logits_per_text, image_features, text_features

    def __call__(self, caption: str, image: Image, boxes: List[Box], image_name: str = None, image_pth=None) -> torch.Tensor:
        if self.expand_position_embedding: 
            original_preprocesses = self.preprocesses
            new_preprocesses = []
            original_position_embeddings = []
            for model_name, model, preprocess in zip(self.clip_models, self.models, self.preprocesses):
                if "RN" in model_name:
                    model_spatial_dim = int((model.visual.attnpool.positional_embedding.shape[0]-1)**0.5)
                    patch_size = model.visual.input_resolution // model_spatial_dim
                    original_positional_embedding = model.visual.attnpool.positional_embedding.clone()
                    model.visual.attnpool.positional_embedding = torch.nn.Parameter(torch.nn.functional.interpolate(
                        model.visual.attnpool.positional_embedding[1:,:].permute(1, 0).view(1, -1, model_spatial_dim, model_spatial_dim),
                        size=(image.height // patch_size, image.width // patch_size),
                        mode='bicubic',
                        align_corners=False
                    ).squeeze(0).permute(1, 2, 0).view(-1, original_positional_embedding.shape[-1]))
                    model.visual.attnpool.positional_embedding = torch.nn.Parameter(torch.cat((
                        original_positional_embedding[:1,:],
                        model.visual.attnpool.positional_embedding
                    ), dim=0))
                    transform = transforms.Compose([
                        transforms.Resize(((image.height // patch_size)*patch_size, (image.width // patch_size)*patch_size), interpolation=Image.BICUBIC),
                        lambda image: image.convert("RGB"),
                        transforms.ToTensor(),
                        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
                    ])
                else:
                    model_spatial_dim = int((model.visual.positional_embedding.shape[0]-1)**0.5)
                    patch_size = model.visual.input_resolution // model_spatial_dim
                    original_positional_embedding = model.visual.positional_embedding.clone()
                    model.visual.positional_embedding = torch.nn.Parameter(torch.nn.functional.interpolate(
                        model.visual.positional_embedding[1:,:].permute(1, 0).view(1, -1, model_spatial_dim, model_spatial_dim),
                        size=(image.height // patch_size, image.width // patch_size),
                        mode='bicubic',
                        align_corners=False
                    ).squeeze(0).permute(1, 2, 0).view(-1, original_positional_embedding.shape[-1]))
                    model.visual.positional_embedding = torch.nn.Parameter(torch.cat((
                        original_positional_embedding[:1,:],
                        model.visual.positional_embedding
                    ), dim=0))
                    transform = transforms.Compose([
                        transforms.Resize(((image.height // patch_size)*patch_size, (image.width // patch_size)*patch_size), interpolation=Image.BICUBIC),
                        lambda image: image.convert("RGB"),
                        transforms.ToTensor(),
                        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
                    ])
                new_preprocesses.append(transform)
                original_position_embeddings.append(original_positional_embedding)
            self.preprocesses = new_preprocesses
        result = super().__call__(caption, image, boxes, image_name, image_pth)
        if self.expand_position_embedding:
            self.preprocesses = original_preprocesses
            for model, model_name, pos_embedding in zip(self.models, self.clip_models, original_position_embeddings):
                if "RN" in model_name:
                    model.visual.attnpool.positional_embedding = torch.nn.Parameter(pos_embedding)
                else:
                    model.visual.positional_embedding = torch.nn.Parameter(pos_embedding)
        return result

