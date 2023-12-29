## Zero-Shot Referring Expression Comprehension on RefCOCO

**Preparing Data**

1.Download [images for RefCOCO/g/+](http://images.cocodataset.org/zips/train2014.zip). Put downloaded dataset(train2014) to eval/rec_zs_test/data/.

2.Download preprocessed data files via `gsutil cp gs://reclip-sanjays/reclip_data.tar.gz` and `cd rec_zs_test`, and then extract the data using `tar -xvzf reclip_data.tar.gz`. 

**Preparing model**

3.Download [SAM](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth) (vit-h), [Alpha-CLIP](https://github.com/SunzeY/AlphaCLIP/blob/main/model-zoo.md) model, and put them in ./eval/rec_zs_test/ckpt.

```
├── eval
│   ├── rec_zs_test
│   │   ├── data
│   │       └── train2014
│   │   ├── reclip_data
│   │       └── refcoco_val.jsonl
│   │       └── refcoco_dets_dict.json
│   │           ...
│   │   ├── ckpt
│   │       └── sam_vit_h_4b8939.pth
│   │       └── grit1m
│   │           └── clip_b16_grit+mim_fultune_4xe.pth
│   │           └── clip_l14_grit+mim_fultune_6xe.pth
│   │   ├── methods
│   │   ├── cache
│   │   ├── output
│   │   ├── main.py
│   │   ├── executor.py
│   │   ├── run.sh
│   │   ├── ...
```

4.run test script.

```
cd eval/rec_zs_test
```
```
bash run.sh
```
or

```
python main.py --input_file reclip_data/refcoco_val.jsonl --image_root ./data/train2014 --method parse --gradcam_alpha 0.5 0.5 --box_representation_method full,blur --box_method_aggregator sum --clip_model ViT-B/16,ViT-L/14 --detector_file reclip_data/refcoco+_dets_dict.json --cache_path ./cache
```
(We recommend using `cache_path` to reduce time to generate mask by SAM for a image repeatedly.`)

For multi-gpus testing, try:

```
bash run_multi_gpus.sh
python cal_acc.py refcoco_val
```


**Acknowledgement**

We test our model based on the wonderful work [ReCLIP](https://github.com/allenai/reclip/tree/main). We simply replace CLIP with Alpha-CLIP; and skip the image-cropping operation.



**Experiment results**

| Method         | RefCOCO |      |      | RefCOCO+ |      |      | RefCOCOg |      |
|----------------|---------|------|------|----------|------|------|----------|------|
|                | Val     | TestA| TestB| Val      | TestA| TestB| Val      | Test |
| CPT [67]       | 32.2    | 36.1 | 30.3 | 31.9     | 35.2 | 28.8 | 36.7     | 36.5 |
| ReCLIP [54]    | 45.8    | 46.1 | 47.1 | 47.9     | 50.1 | 45.1 | 59.3     | 59.0 |
| Red Circle [52]| 49.8    | 58.6 | 39.9 | 55.3     | 63.9 | 45.4 | 59.4     | 58.9 |
| Alpha-CLIP     | 55.7    | 61.1 | 50.3 | 55.6     | 62.7 | 46.4 | 61.2     | 62.0 |

