import os
import clip
import json
import argparse
import ruamel.yaml as yaml

from PIL import Image
import torch
import torchvision.transforms as transforms
from tqdm import tqdm

from albef.utils import *
from executor import AlbefExecutor

parser = argparse.ArgumentParser()
parser.add_argument("--input_path", type=str, help="Path to input JSON file")
parser.add_argument("--image_root", type=str, help="Path to directory containing images")
parser.add_argument("--albef_path", type=str, default=None, help="Path to ALBEF model/config/etc. if the goal is to use ALBEF")
parser.add_argument("--albef_itc", action="store_true", help="Use ITC output of ALBEF")
parser.add_argument("--clip_model", type=str, help="CLIP model to use")
parser.add_argument("--gpu", type=int, default=-1, help="Which gpu to use")
parser.add_argument("--batch_size", type=int, default=32, help="Batch size for running CLIP")

args = parser.parse_args()

if args.albef_path is not None:
    executor = AlbefExecutor(checkpoint_path = os.path.join(args.albef_path, "checkpoint.pth"), config_path = os.path.join(args.albef_path, "config.yaml"), device = "cpu" if args.gpu < 0 else "cuda:"+str(args.gpu))
    model = executor.models[0]
    preprocess = executor.preprocesses[0]
    model = model.eval()
else:
    model, preprocess = clip.load(args.clip_model, jit=False, device="cuda:"+str(args.gpu))
    preprocess.transforms[0] == transforms.Resize((model.visual.input_resolution, model.visual.input_resolution), transforms.InterpolationMode.BICUBIC)
    model = model.eval()
input_file = open(args.input_path)
data = json.load(input_file)
input_file.close()
correct = 0
for i in tqdm(range(0, len(data), args.batch_size)):
    batch_images = []
    batch_text = []
    for datum in data[i:min(i+args.batch_size, len(data))]:
        img = Image.open(os.path.join(args.image_root, datum["image_filename"])).convert('RGB')
        batch_images.append(preprocess(img))
        if "text2" in datum:
            if args.albef_path is None:
                datum["text1"] = "a photo of "+datum["text1"]
                datum["text2"] = "a photo of "+datum["text2"]
            batch_text.append(datum["text1"])
            batch_text.append(datum["text2"])
        else:
            img2 = Image.open(os.path.join(args.image_root, datum["image_filename2"])).convert('RGB')
            batch_images.append(preprocess(img2))
            batch_text.append(datum["text1"])
    batch_images = torch.stack(batch_images).to("cuda:"+str(args.gpu))
    if args.albef_path is None:
        batch_text = clip.tokenize(batch_text).to("cuda:"+str(args.gpu))
    else:
        modified_text = [pre_caption(txt, executor.max_words) for txt in batch_text]
        batch_text = executor.tokenizer(modified_text, padding='longest', return_tensors="pt")
        for key in batch_text:
            batch_text[key] = batch_text[key].to(batch_images.device)

    with torch.no_grad():
        if args.albef_path is None:
            logits_per_image, logits_per_text = model(batch_images, batch_text)
        else:
            if not args.albef_itc:
                if batch_images.shape[0]*2 == batch_text.input_ids.shape[0]:
                    batch_images = batch_images.unsqueeze(1).repeat(1, 2, 1, 1, 1).view(batch_images.shape[0]*2, batch_images.shape[1], batch_images.shape[2], batch_images.shape[3])
                else:
                    assert batch_images.shape[0] ==2*batch_text.input_ids.shape[0]
                    batch_text.input_ids = batch_text.input_ids.unsqueeze(1).repeat(1, 2, 1).view(batch_images.shape[0], -1)
                    batch_text.attention_mask = batch_text.attention_mask.unsqueeze(1).repeat(1, 2, 1).view(batch_images.shape[0], -1)
                image_embeds = model.visual_encoder(batch_images)
                image_atts = torch.ones(image_embeds.size()[:-1],dtype=torch.long).to(batch_images.device)
                output = model.text_encoder(
                    batch_text.input_ids,
                    attention_mask = batch_text.attention_mask,
                    encoder_hidden_states = image_embeds,
                    encoder_attention_mask = image_atts,      
                    return_dict = True,
                )
                vl_embeddings = output.last_hidden_state[:,0,:]
                vl_output = model.itm_head(vl_embeddings)
                logits_per_image = vl_output[:,1:2].view(-1, 2)
            else:
                image_embeds = model.visual_encoder(batch_images)
                image_feat = torch.nn.functional.normalize(model.vision_proj(image_embeds[:,0,:]),dim=-1) 
                text_output = model.text_encoder(batch_text.input_ids, attention_mask = batch_text.attention_mask,                 
                                                 return_dict = True, mode = 'text')            
                text_embeds = text_output.last_hidden_state
                text_feat = torch.nn.functional.normalize(model.text_proj(text_embeds[:,0,:]),dim=-1)     
                sim = image_feat@text_feat.t()/model.temp
                logits_per_image = sim
    if args.albef_path is None or args.albef_itc:
        if logits_per_image.shape[0]*2 == logits_per_image.shape[1]:
            for j in range(logits_per_image.shape[0]):
                correct += 1 if logits_per_image[j,2*j].item() > logits_per_image[j,2*j+1].item() else 0
        else:
            assert logits_per_image.shape[0] == 2*logits_per_image.shape[1]
            for j in range(logits_per_image.shape[1]):
                correct += 1 if logits_per_image[2*j,j].item() > logits_per_image[2*j+1,j].item() else 0
    else:
        correct += (logits_per_image[:,0] > logits_per_image[:,1]).long().sum().item()

print("Accuracy:", correct/len(data))
