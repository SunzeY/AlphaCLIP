import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid
import numpy as np
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from torch.utils.data import Dataset, DataLoader
from pycocotools import mask as mask_utils
from PIL import Image
import math
import collections
import types
mask_torch = None

def rewrited_forward(self, pixel_values: torch.FloatTensor) -> torch.Tensor:
    # print("[Warning] using rewrited alpha forword")
    global mask_torch
    # mask_torch = None
    batch_size = pixel_values.shape[0]
    patch_embeds = self.patch_embedding(pixel_values)  # shape = [*, width, grid, grid]
    if mask_torch is None:
        print("[Warning] no mask specified!")
        alpha = torch.ones_like((pixel_values[:, [0], :, :])) * 1.9231
    else:
        alpha = mask_torch
    patch_embeds = patch_embeds + self.patch_embedding_alpha(alpha)
    patch_embeds = patch_embeds.flatten(2).transpose(1, 2)

    class_embeds = self.class_embedding.expand(batch_size, 1, -1)
    embeddings = torch.cat([class_embeds, patch_embeds], dim=1)
    embeddings = embeddings + self.position_embedding(self.position_ids)
    return embeddings


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


# Custom dataset class
class CustomDataset(Dataset):
    def __init__(self, questions, image_folder, tokenizer, image_processor, model_config):
        self.questions = questions
        self.image_folder = image_folder
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.model_config = model_config

    def __getitem__(self, index):
        line = self.questions[index]
        image_file = line["image"]
        qs = line["text"]
        if self.model_config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        image = Image.open(os.path.join(self.image_folder, image_file)).convert('RGB')
        image_tensor = process_images([image], self.image_processor, self.model_config)[0]

        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')
        
        wb_mask = Image.fromarray(np.expand_dims(mask_utils.decode(line['focus_region']), axis=-1)[:, :, [0, 0, 0]] * 255)

        if wb_mask._size != image._size: # bug case, set to all one
            wb_mask = Image.fromarray((np.ones_like(np.array(image)) * 255).astype(np.uint8))
        # wb_mask = Image.fromarray((np.ones_like(np.array(image)) * 255).astype(np.uint8))
        # image.save(f"test_image_{index}.png")
        # wb_mask.save(f"test_mask_{index}.png")
        mask_tensor = process_images([wb_mask], self.image_processor, self.model_config)[0]
        masks = mask_tensor.sum(dim=0)
        mask_tensor = (masks > 6) * 1.9231 + (masks <= 6) * (-1.9231)

        return input_ids, image_tensor, mask_tensor

    def __len__(self):
        return len(self.questions)


# DataLoader
def create_data_loader(questions, image_folder, tokenizer, image_processor, model_config, batch_size=1, num_workers=4):
    assert batch_size == 1, "batch_size must be 1"
    dataset = CustomDataset(questions, image_folder, tokenizer, image_processor, model_config)
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=4, shuffle=False)
    return data_loader


def eval_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)
    
    visual_encoder = model.model.vision_tower.vision_tower.vision_model
    visual_encoder.embeddings.patch_embedding_alpha = torch.nn.Conv2d(in_channels=1,
                                                        out_channels=visual_encoder.embeddings.patch_embedding.out_channels, 
                                                        kernel_size=visual_encoder.embeddings.patch_embedding.kernel_size, 
                                                        stride=visual_encoder.embeddings.patch_embedding.stride, 
                                                        bias=False)
    visual_encoder.embeddings.forward = types.MethodType(rewrited_forward, visual_encoder.embeddings)
    state_dict = torch.load('clip_l14@336_grit1m_fultune_8xe.pth')
    converted_dict = collections.OrderedDict()
    for k, v in state_dict.items():
        if 'transformer.resblocks' in k:
            new_key = k.replace('transformer.resblocks', 'encoder.layers').replace('attn', 'self_attn').replace('ln_1', 'layer_norm1').replace('ln_2', 'layer_norm2') \
                    .replace('c_fc', 'fc1').replace('c_proj', 'fc2')
            if ('self_attn' in new_key) and ('out' not in new_key): # split qkv attn
                if 'weight' in new_key :
                    converted_dict[new_key.replace('in_proj', 'q_proj')] = v[:1024, :]
                    converted_dict[new_key.replace('in_proj', 'k_proj')] = v[1024:2048, :]
                    converted_dict[new_key.replace('in_proj', 'v_proj')] = v[2048:, :]
                else:
                    assert 'bias' in new_key
                    converted_dict[new_key.replace('in_proj', 'q_proj')] = v[:1024]
                    converted_dict[new_key.replace('in_proj', 'k_proj')] = v[1024:2048]
                    converted_dict[new_key.replace('in_proj', 'v_proj')] = v[2048:]
            else:
                converted_dict[new_key] = v
        else:
            new_key = k.replace('class_embedding', 'embeddings.class_embedding') \
                    .replace('conv1.weight', 'embeddings.patch_embedding.weight') \
                    .replace('positional_embedding', 'embeddings.position_embedding.weight') \
                    .replace('conv1_alpha.weight', 'embeddings.patch_embedding_alpha.weight') \
                    .replace('ln_pre.weight', 'pre_layrnorm.weight') \
                    .replace('ln_pre.bias', 'pre_layrnorm.bias') \
                    .replace('ln_post.weight', 'post_layernorm.weight') \
                    .replace('ln_post.bias', 'post_layernorm.bias')
            converted_dict[new_key] = v

    visual_encoder.load_state_dict(converted_dict, strict=False)
    visual_encoder = visual_encoder.half().cuda()

    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")

    if 'plain' in model_name and 'finetune' not in model_name.lower() and 'mmtag' not in args.conv_mode:
        args.conv_mode = args.conv_mode + '_mmtag'
        print(f'It seems that this is a plain model, but it is not using a mmtag prompt, auto switching to {args.conv_mode}.')

    data_loader = create_data_loader(questions, args.image_folder, tokenizer, image_processor, model.config)

    for (input_ids, image_tensor, mask_tensor), line in tqdm(zip(data_loader, questions), total=len(questions)):
        idx = line["question_id"]
        cur_prompt = line["text"]

        stop_str = conv_templates[args.conv_mode].sep if conv_templates[args.conv_mode].sep_style != SeparatorStyle.TWO else conv_templates[args.conv_mode].sep2
        input_ids = input_ids.to(device='cuda', non_blocking=True)

        with torch.inference_mode():
            global mask_torch
            mask_torch = mask_tensor.to(dtype=torch.float16, device='cuda', non_blocking=True).unsqueeze(dim=1)
            output_ids = model.generate(
                input_ids,
                images=image_tensor.to(dtype=torch.float16, device='cuda', non_blocking=True),
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                max_new_tokens=128,
                use_cache=True)

        input_token_len = input_ids.shape[1]
        n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
        outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
        outputs = outputs.strip()
        if outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)]
        outputs = outputs.strip()

        ans_id = shortuuid.uuid()
        ans_file.write(json.dumps({"question_id": idx,
                                   "prompt": cur_prompt,
                                   "text": outputs,
                                   "answer_id": ans_id,
                                   "model_id": model_name,
                                   "label": line["label"] if "label" in line.keys() else None,
                                   "metadata": {}}) + "\n")
        # ans_file.flush()
    ans_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    args = parser.parse_args()

    eval_model(args)