# <img src="img/alpha_icon.png" style="vertical-align: -10px;" :height="40px" width="40px"> Alpha-CLIP
This repository is the official implementation of
**[Alpha-CLIP: A CLIP model focusing on wherever you want](https://arxiv.org/abs/2307.04725)**
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
<p style="font-size: 0.8em; margin-top: -1em">*Equal Contribution</p>

<!-- [Arxiv Report](https://arxiv.org/abs/2307.04725) | [Project Page](https://animatediff.github.io/) -->
[![Arxiv Report](https://img.shields.io/badge/arXiv-2307.04725-b31b1b.svg)](https://arxiv.org/abs/2307.04725)
[![Project Page](https://img.shields.io/badge/Project-Website-green)](https://animatediff.github.io/)
[![Open in OpenXLab](https://cdn-static.openxlab.org.cn/app-center/openxlab_app.svg)](https://openxlab.org.cn/apps/detail/Masbfca/AnimateDiff) with LLM
[![Open in OpenXLab](https://cdn-static.openxlab.org.cn/app-center/openxlab_app.svg)](https://openxlab.org.cn/apps/detail/Masbfca/AnimateDiff) with SD
[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-yellow)](https://huggingface.co/spaces/guoyww/AnimateDiff) with LLM
[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-yellow)](https://huggingface.co/spaces/guoyww/AnimateDiff) with SD

## 📜 News
[2023/11/20] The [paper]([ShareGPT4V.pdf](https://arxiv.org/pdf/2311.12793.pdf)) and [project page](https://ShareGPT4V.github.io/) are released!

## 💡 Highlights
- 🔥 **3.93%** improved zero-shot ImageNet classification accuracy when providing foreground alpha-map.
- 🔥 **Plug-in and play** with region focus in **any work** that use CLIP vision encoder.
- 🔥 **A strong visual encoder** as vasatile tool when foreground mask is available.

## 👨‍💻 Todo
- [ ] Training and evaluation code for Alpha-CLIP
- [x] Web demo and local demo of Alpha-CLIP
- [x] Usage example notebook of Alpha-CLIP
- [x] Checkpoints of Alpha-CLIP

## 🛠️Usage

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
download model from [model-zoo](https://github.com/SunzeY/AlphaCLIP/blob/main/model-zoo.md) and place it under `checkpoints`.

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

Usage examples are available

* visualization of attention map: [notebook](https://github.com/SunzeY/AlphaCLIP/blob/main/notebooks/attn_visual.ipynb)
* Alpha-CLIP used in BLIP-Diffusion: [notebook](https://github.com/SunzeY/AlphaCLIP/blob/main/notebooks/blip_diffusion.ipynb)

## ❤️ Acknowledgments
- [CLIP](https://github.com/openai/CLIP): The codebase we built upon. Thanks for their wonderful work.
- [LAVIS](https://github.com/salesforce/LAVIS): The amazing open-sourced multimodality learning codebase, where we test Alpha-CLIP in [BLIP-2](https://github.com/salesforce/LAVIS/tree/main/projects/blip2) and [BLIP-Diffusion](https://github.com/salesforce/LAVIS/tree/main/projects/blip-diffusion).
- [Point-E](https://github.com/openai/point-e): Wonderful point-cloud generation model, where we test Alpha-CLIP for 3D generation task.
- [LLaVA](https://github.com/haotian-liu/LLaVA): Wounderful MLLM that use CLIP as visual bacbone where we test the effectiveness of Alpha-CLIP.

## ✒️ Citation
If you find our work helpful for your research, please consider giving a star ⭐ and citation 📝
```bibtex

```

## License
![Code License](https://img.shields.io/badge/Code%20License-Apache_2.0-green.svg) ![Data License](https://img.shields.io/badge/Data%20License-CC%20By%20NC%204.0-red.svg) **Usage and License Notices**: The data and checkpoint is intended and licensed for research use only. They are also restricted to uses that follow the license agreement of CLIP. The dataset is CC BY NC 4.0 (allowing only non-commercial use) and models trained using the dataset should not be used outside of research purposes.
