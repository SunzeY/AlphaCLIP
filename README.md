# Alpha-CLIP

## prepare alpha CLIP
```shell
pip install -e .
```
install loralib
```shell
pip install loralib
```

## how to use
download model and place it under `checkpoints`.

```python
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
