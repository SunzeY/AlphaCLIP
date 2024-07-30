# Training Alpha-CLIP (on GRIT-20M only)

## Data Preparation
1. Download [GRIT-20M](https://huggingface.co/datasets/zzliang/GRIT).
2. use `train/data_prepare/sam_grit.py` to generate mask given boxes with [SAM](https://github.com/facebookresearch/segment-anything).
3. modify training dataloader `get_file` function to adapt to your own data path.

## Testing 
We test Alpha-CLIP on two kinds of settings during training.
1. General image recognition ability measured by ImageNet-S Classification accuracy. You need to prepare [ImageNet-S](https://github.com/LUSSeg/ImageNet-S) dataset. Download our processed annotations here ([imagenet_919.json](https://huggingface.co/datasets/Zery/MaskImageNet/tree/main)) The folder should be structured like this or adjust path in dataloader.
```
├── imagenet-s
│   ├── data
│   │   ├── imagenet_919.json
│   │   └── ImageNetS919
│   │       └── validation
```
3. (Optional) Region level recognition ability measured by COCO / LVIS Classification accuracy (with full image as input). You need to prepare [COCO](https://cocodataset.org/#home)-2017 dataset accordingly.

## Training
We use SLURM for multi-nodes training.
```bash
bash train_slurm.sh
```
you may need to adjust `self.batch_size` or use LoRA in `train_grit_1m.py` to adapte to your training resources. We only claim reproducibility using **the same batchsize** and **full model finetuning** in our paper.

# Training Alpha-CLIP (with additional MaskImageNet)
1. download data annotations [here](https://huggingface.co/datasets/Zery/MaskImageNet)
2. download full ImageNet-21K dataset
3. train with `train_grit_1m+mim.py`
