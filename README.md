# Alpha-CLIP

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

## ❤️ Acknowledgments
- [CLIP](https://github.com/openai/CLIP): The codebase we built upon. Thanks for their wonderful work.
- [LAVIS](https://github.com/salesforce/LAVIS): The amazing open-sourced multimodality learning codebase, where we test Alpha-CLIP in [BLIP-2](https://github.com/salesforce/LAVIS/tree/main/projects/blip2) and [BLIP-Diffusion](https://github.com/salesforce/LAVIS/tree/main/projects/blip-diffusion).
- [Point-E](https://github.com/openai/point-e): Wonderful point-cloud generation model, where we test Alpha-CLIP for 3D generation task.
- [LLaVA](https://github.com/haotian-liu/LLaVA) Wounderful MLLM that use CLIP as visual bacbone where we test the effectiveness of Alpha-CLIP.

## ✒️ Citation
If you find our work helpful for your research, please consider giving a star ⭐ and citation 📝
```bibtex

```

## License
![Code License](https://img.shields.io/badge/Code%20License-Apache_2.0-green.svg) ![Data License](https://img.shields.io/badge/Data%20License-CC%20By%20NC%204.0-red.svg) **Usage and License Notices**: The data and checkpoint is intended and licensed for research use only. They are also restricted to uses that follow the license agreement of CLIP. The dataset is CC BY NC 4.0 (allowing only non-commercial use) and models trained using the dataset should not be used outside of research purposes.
