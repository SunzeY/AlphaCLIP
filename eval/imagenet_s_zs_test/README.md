# Alpha-CLIP evaluation
## Zero-Shot Classification on ImageNet-S

1.prepare [imagenet-s](https://github.com/LUSSeg/ImageNet-S) dataset, only `validation` raw image is needed.

2.download [imagenet_919.json](https://download.openxlab.org.cn/models/SunzeY/AlphaCLIP/weight/imagenet_919.json) we provide as data annotation (generated from imagenet-s annotation). The folder should be structured like

```
├── imagenet_s_zs_test
│   ├── data
│   │   ├── imagenet_919.json
│   │   └── ImageNetS919
│   │       └── validation
```

3.run test script.

```
cd eval/imagenet_s_zs_test
python imagenet_s_zs_test.py
```
