# <img src="img/alpha_icon.png" style="vertical-align: -10px;" :height="40px" width="40px"> Alpha-CLIP
This repository is the official implementation of AlphaCLIP

**[Alpha-CLIP: A CLIP Model Focusing on Wherever You Want](https://arxiv.org/abs/2312.03818)**
</br>
[Zeyi Sun](https://github.com/SunzeY)\*,
[Ye Fang](https://github.com/Aleafy)\*,
[Tong Wu](https://wutong16.github.io/),
[Pan Zhang](https://panzhang0212.github.io/),
[Yuhang Zang](https://yuhangzang.github.io/),
[Shu Kong](https://aimerykong.github.io/),
[Yuanjun Xiong](http://yjxiong.me/),
[Dahua Lin](http://dahua.site/),
[Jiaqi Wang](https://myownskyw7.github.io/)
<p style="font-size: 0.6em; margin-top: -1em">*Equal Contribution</p>
<p align="center">
<a href="https://arxiv.org/abs/2312.03818"><img src="https://img.shields.io/badge/arXiv-Paper-<color>"></a>
<a href="https://aleafy.github.io/alpha-clip"><img src="https://img.shields.io/badge/Project-Website-red"></a>
</p>

Demo `Alpha-CLIP` with `Stable Diffusion`: 
[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-yellow)](https://huggingface.co/spaces/Zery/Alpha_CLIP_ImgVar)
[![Open in OpenXLab](https://cdn-static.openxlab.org.cn/app-center/openxlab_app.svg)](https://openxlab.org.cn/apps/detail/SunzeY/Alpha-CLIP_Image_Var1) 


Demo `Alpha-CLIP` with `LLaVA`:
[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-yellow)](https://huggingface.co/spaces/Zery/Alpha-CLIP_LLaVA-1.5)
[![Open in OpenXLab](https://cdn-static.openxlab.org.cn/app-center/openxlab_app.svg)](https://openxlab.org.cn/apps/detail/SunzeY/Alpha-CLIP_with_LLM) 


## 📜 News
🚀 [2024/3/4] CLIP-L/14@336px finetuned on GRIT-20M is available, checkout [model-zoo](https://github.com/SunzeY/AlphaCLIP/blob/main/model-zoo.md)!

🚀 [2024/2/27] Our paper Alpha-CLIP is accepted by CVPR'24!

🚀 [2024/1/2] Zero-shot testing code for [Imagenet-S Classification](https://github.com/SunzeY/AlphaCLIP/tree/eval-dev/eval/imagenet_s_zs_test) and [Referring Expression Comprehension](https://github.com/SunzeY/AlphaCLIP/tree/eval-dev/eval/rec_zs_test) are released!

🚀 [2023/12/27] Web [demo](https://huggingface.co/spaces/Zery/Alpha-CLIP_LLaVA-1.5) and local [demo](https://github.com/SunzeY/AlphaCLIP/tree/main/demo/with_llm) of Alpha-CLIP with LLaVA are released!

🚀 [2023/12/7] Web [demo](https://huggingface.co/spaces/Zery/Alpha_CLIP_ImgVar) and local [demo](https://github.com/SunzeY/AlphaCLIP/tree/main/demo/with_diffusion) of Alpha-CLIP with Stable Diffusion are released!

🚀 [2023/12/7] The [paper](https://arxiv.org/abs/2312.03818) and [project page](https://aleafy.github.io/alpha-clip) are released!

## 💡 Highlights
- 🔥 **3.93%** improved zero-shot ImageNet classification accuracy when providing foreground alpha-map.
- 🔥 **Plug-in and play** with region focus in **any work** that use CLIP vision encoder.
- 🔥 **A strong visual encoder** as versatile tool when foreground mask is available.

## 👨‍💻 Todo
- [ ] Training code for Alpha-CLIP based on Open-CLIP
- [x] Evaluation code for Alpha-CLIP
- [x] Zero-shot evaluation for Imagenet-S Classification and REC tasks.
- [x] Web demo and local demo of Alpha-CLIP with LLaVA
- [x] Web demo and local demo of Alpha-CLIP with Stable Diffusion
- [x] Usage example notebook of Alpha-CLIP
- [x] Checkpoints of Alpha-CLIP

## 🛠️ Usage

### Installation
our model is based on [CLIP](https://github.com/openai/CLIP), please first prepare environment for CLIP, then directly install Alpha-CLIP.

```shell
pip install -e .
```

install loralib

```shell
pip install loralib
```

### how to use
Download model from [model-zoo](https://github.com/SunzeY/AlphaCLIP/blob/main/model-zoo.md) and place it under `checkpoints`.

```python
import alpha_clip
alpha_clip.load("ViT-B/16", alpha_vision_ckpt_pth="checkpoints/clip_b16_grit1m_fultune_8xe.pth", device="cpu"), 
image_features = model.visual(image, alpha)
```
`alpha` need to be normalized via transforms when using `binary_mask` in (0, 1)

```python
mask_transform = transforms.Compose([
    transforms.ToTensor(), 
    transforms.Resize((224, 224)),
    transforms.Normalize(0.5, 0.26)
])
alpha = mask_transform(binary_mask * 255)
```



**Zero-shot Prediction**

```python
import torch
import alpha_clip
from PIL import Image
import numpy as np
from torchvision import transforms

# load model and prepare mask transform
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = alpha_clip.load("ViT-L/14", alpha_vision_ckpt_pth="./checkpoints/clip_l14_grit20m_fultune_2xe.pth", device=device)  # change to your own ckpt path
mask_transform = transforms.Compose([
    transforms.ToTensor(), 
    transforms.Resize((224, 224)), # change to (336,336) when using ViT-L/14@336px
    transforms.Normalize(0.5, 0.26)
])

# prepare image and mask
img_pth = './examples/image.png'
mask_pth = './examples/dress_mask.png' # image-type mask

image = Image.open(img_pth).convert('RGB')
mask = np.array(Image.open(mask_pth)) 
# get `binary_mask` array (2-dimensional bool matrix)
if len(mask.shape) == 2: binary_mask = (mask == 255)
if len(mask.shape) == 3: binary_mask = (mask[:, :, 0] == 255)

alpha = mask_transform((binary_mask * 255).astype(np.uint8))
alpha = alpha.half().cuda().unsqueeze(dim=0)

# calculate image and text features
image = preprocess(image).unsqueeze(0).half().to(device)
text = alpha_clip.tokenize(["a goegously dressed woman", "a purple sleeveness dress", "bouquet of pink flowers"]).to(device)

with torch.no_grad():
    image_features = model.visual(image, alpha)
    text_features = model.encode_text(text)

# normalize
image_features = image_features / image_features.norm(dim=-1, keepdim=True)
text_features = text_features / text_features.norm(dim=-1, keepdim=True)

## print the result
similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
print("Label probs:", similarity.cpu().numpy()) # prints: [[9.388e-05 9.995e-01 2.415e-04]]
```

Note: Using `.half()` for tensor or `.float()` for model to maintain type consistency.



More usage examples are available：

* Visualization of attention map: [notebook](https://github.com/SunzeY/AlphaCLIP/blob/main/notebooks/attn_visual.ipynb)
* Alpha-CLIP used in BLIP-Diffusion: [notebook](https://github.com/SunzeY/AlphaCLIP/blob/main/notebooks/blip_diffusion.ipynb)
* Alpha-CLIP used in SD_ImageVar: [demo](https://github.com/SunzeY/AlphaCLIP/tree/main/demo/with_diffusion)
* Alpha-CLIP used in LLaVA-1.5: [code](https://github.com/SunzeY/AlphaCLIP/tree/main/demo/with_llm)  [demo](https://huggingface.co/spaces/Zery/Alpha-CLIP_LLaVA-1.5) 
* Alpha-CLIP evaluation code for Image Recognition: [code](https://github.com/SunzeY/AlphaCLIP/tree/eval-dev/eval)  

##   ⭐ Demos
<p align="center"> <a>  
<img src="./img/demo1.gif"  width="900" />
</a> </p>



## ❤️ Acknowledgments
- [CLIP](https://github.com/openai/CLIP): The codebase we built upon. Thanks for their wonderful work.
- [LAVIS](https://github.com/salesforce/LAVIS): The amazing open-sourced multimodality learning codebase, where we test Alpha-CLIP in [BLIP-2](https://github.com/salesforce/LAVIS/tree/main/projects/blip2) and [BLIP-Diffusion](https://github.com/salesforce/LAVIS/tree/main/projects/blip-diffusion).
- [Point-E](https://github.com/openai/point-e): Wonderful point-cloud generation model, where we test Alpha-CLIP for 3D generation task.
- [LLaVA](https://github.com/haotian-liu/LLaVA): Wounderful MLLM that use CLIP as visual bacbone where we test the effectiveness of Alpha-CLIP.

## ✒️ Citation
If you find our work helpful for your research, please consider giving a star ⭐ and citation 📝
```bibtex
@misc{sun2023alphaclip,
      title={Alpha-CLIP: A CLIP Model Focusing on Wherever You Want}, 
      author={Zeyi Sun and Ye Fang and Tong Wu and Pan Zhang and Yuhang Zang and Shu Kong and Yuanjun Xiong and Dahua Lin and Jiaqi Wang},
      year={2023},
      eprint={2312.03818},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

## License
![Code License](https://img.shields.io/badge/Code%20License-Apache_2.0-green.svg) ![Data License](https://img.shields.io/badge/Data%20License-CC%20By%20NC%204.0-red.svg) **Usage and License Notices**: The data and checkpoint is intended and licensed for research use only. They are also restricted to uses that follow the license agreement of CLIP. The dataset is CC BY NC 4.0 (allowing only non-commercial use) and models trained using the dataset should not be used outside of research purposes.
