import gradio as gr
import torch
from PIL import Image
from torchvision import transforms
from transformers import AutoModelForZeroShotImageClassification
from diffusers import StableDiffusionImageVariationPipeline
import os
import collections
import types
import copy
import numpy as np
import wget

os.environ["no_proxy"] = "localhost"

mask_torch = None

mask_transform = transforms.Compose([
    transforms.ToTensor(), 
    transforms.Resize((224, 224)),
    transforms.Normalize(0.5, 0.26)
])

def rewrited_forward(self, pixel_values: torch.FloatTensor) -> torch.Tensor:
    print("[Warning] using rewrited alpha forword")
    global mask_torch
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

class ImageMask(gr.components.Image):
    """
    Sets: source="canvas", tool="sketch"
    """

    is_template = True
    def __init__(self, **kwargs):
        super().__init__(source="upload", tool="sketch", interactive=True, **kwargs)

    def preprocess(self, x):
        return super().preprocess(x)

@torch.no_grad()
def main(
    input_im,
    task,
    scale=3.0,
    n_samples=4,
    steps=25,
    seed=0,
    ):
    
    if "Ori CLIP" == task: # different samplers
        pipe.image_encoder.vision_model = ori_visual_encoder
    else: 
        pipe.image_encoder.vision_model = visual_encoder
    generator = torch.Generator(device=device).manual_seed(int(seed))

    tform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(
        (224, 224),
        interpolation=transforms.InterpolationMode.BICUBIC,
        antialias=False,
        ),
        transforms.Normalize(
          [0.48145466, 0.4578275, 0.40821073],
          [0.26862954, 0.26130258, 0.27577711]),
    ])

    inp = tform(input_im['image']).to(device)
    mask_ori = input_im['mask'] # different mask strategy
    mask = np.array(mask_ori)[:,:,0:1]
    if "With mask" == task: # highlight
        mask = (mask[:, :, 0] > 0)
    if "No mask" == task: # All one
        mask = (mask[:, :, 0] > -1)

    mask = mask_transform((mask * 255).astype(np.uint8))
    mask = mask.cuda().unsqueeze(dim=0)
    global mask_torch
    mask_torch = mask

    images_list = pipe(
        inp.tile(n_samples, 1, 1, 1),
        guidance_scale=scale,
        num_inference_steps=steps,
        generator=generator,
        )

    images = []
    for i, image in enumerate(images_list["images"]):
        if(images_list["nsfw_content_detected"][i]):
            safe_image = Image.open(r"unsafe.png")
            images.append(safe_image)
        else:
            images.append(image)
    return images

article = \
"""
## How does Stable Diffusion Image Variation work?
The normal Stable Diffusion model is trained to be conditioned on text input. This version has had the original text encoder (from CLIP) removed, and replaced with
the CLIP _image_ encoder instead. So instead of generating images based a text input, images are generated to match CLIP's embedding of the image.
This creates images which have the same rough style and content, but different details, in particular the composition is generally quite different.
This is a totally different approach to the img2img script of the original Stable Diffusion and gives very different results.
The model was fine tuned on the [LAION aethetics v2 6+ dataset](https://laion.ai/blog/laion-aesthetics/) to accept the new conditioning.
Training was done on 8xA100 GPUs on [Lambda GPU Cloud](https://lambdalabs.com/service/gpu-cloud).
More details are on the [model card](https://huggingface.co/lambdalabs/sd-image-variations-diffusers).
"""

device = "cuda" if torch.cuda.is_available() else "cpu"
pipe = StableDiffusionImageVariationPipeline.from_pretrained(
    "lambdalabs/sd-image-variations-diffusers",
    )
pipe = pipe.to(device)
ori_visual_encoder = copy.deepcopy(pipe.image_encoder.vision_model)

visual_encoder = pipe.image_encoder.vision_model
visual_encoder.embeddings.patch_embedding_alpha = torch.nn.Conv2d(in_channels=1,
                                                        out_channels=visual_encoder.embeddings.patch_embedding.out_channels, 
                                                        kernel_size=visual_encoder.embeddings.patch_embedding.kernel_size, 
                                                        stride=visual_encoder.embeddings.patch_embedding.stride, 
                                                        bias=False)
visual_encoder.embeddings.forward = types.MethodType(rewrited_forward, visual_encoder.embeddings)
filename = "clip_l14_grit20m_fultune_2xe.pth"
if not os.path.exists(filename):
    filename = wget.download("https://download.openxlab.org.cn/models/SunzeY/AlphaCLIP/weight/clip_l14_grit20m_fultune_2xe.pth")
print(filename)
state_dict = torch.load(filename)
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
visual_encoder = visual_encoder.cuda()
inputs = [
    ImageMask(label="[Stroke] Draw on Image",type="pil"), gr.inputs.Radio(choices=["With mask", "No mask", "Ori CLIP"], type="value", label="Interative Mode"),
    gr.Slider(0, 25, value=3, step=1, label="Guidance scale"),
    gr.Slider(1, 4, value=1, step=1, label="Number images"),
    gr.Slider(5, 50, value=25, step=5, label="Steps"),
    gr.Number(0, label="Seed", precision=0)
]
output = gr.Gallery(label="Generated variations")
output.style(grid=2)

examples = [
    ["examples/vermeer.jpg", "Ori CLIP", 3, 1, 25, 0],
    ["examples/matisse.jpg", "Ori CLIP", 3, 1, 25, 0],
]

with gr.Blocks() as demo:
    title_markdown = ("""
    # 🌟 [2D Image Varation] Alpha-CLIP with Stable-Diffusion
    ### 🔊 Notice: This demo only involve Alpha-CLIP used in Stable-Diffusion, for other usage please checkout our work!
    [[Project Page](https://sharegpt4v.github.io/)] [[Code](https://github.com/InternLM/InternLM-XComposer/tree/main/projects/ShareGPT4V)] | 📚 [[Paper](https://arxiv.org/pdf/2311.12793.pdf)]
    ### 🔧 Usage
    * Use `With mask` to use Alpha-CLIP, draw on image directly with stroke to specify region that need focuing.
    * Use `No mask` to use Alpha-CLIP to focus on every part of the image with all 1 alpha-map
    * Use `Ori CLIP` to use original CLIP for comparision.            
    """)
    tos_markdown = ("""
    ### Terms of use
    By using this service, users are required to agree to the following terms:
    The service is a research preview intended for non-commercial use only. It only provides limited safety measures and may generate offensive content. It must not be used for any illegal, harmful, violent, racist, or sexual purposes.
    For an optimal experience, please use desktop computers for this demo, as mobile devices may compromise its quality.
    """)
    learn_more_markdown = ("""
    ### License
    The service is a research preview intended for non-commercial use only, subject to the model [License](https://github.com/openai/CLIP/blob/main/LICENSE) of CLIP and [lambda-diffusers](https://github.com/LambdaLabsML/lambda-diffusers), [Terms of Use](https://openai.com/policies/terms-of-use) of the data generated by OpenAI, and [Privacy Practices](https://chrome.google.com/webstore/detail/sharegpt-share-your-chatg/daiacboceoaocpibfodeljbdfacokfjb) of ShareGPT. Please contact us if you find any potential violation.
    """)
    ack_markdown = ("""
    ### Acknowledgement
    The template for this web demo is built based on [SD_ImageVar](https://huggingface.co/spaces/lambdalabs/stable-diffusion-image-variations), and we are very grateful to [lambda-diffusers](https://github.com/LambdaLabsML/lambda-diffusers) for their open source contributions to the community!
    """)

    gr.Markdown(title_markdown)
    gr.Interface(
        fn=main,
        article=article,
        inputs=inputs,
        outputs=output,
        examples=examples,
        )
    gr.Markdown(tos_markdown)
    gr.Markdown(learn_more_markdown)
    gr.Markdown(ack_markdown)
demo.launch(share=True)
