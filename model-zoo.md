# Model Zoo

Models are also available at [![Open in OpenXLab](https://cdn-static.openxlab.org.cn/header/openxlab_models.svg)](https://openxlab.org.cn/models/detail/SunzeY/AlphaCLIP)

## CLIP

These model trained with original CLIP text-encoder from orignal [CLIP codebase]((https://github.com/openai/CLIP)) from OpenAI. The text-encoder stays fixed during the whole training process.

test metric is classification accuracy on Imagenet-S dataset. `()` is improved acc compared to original CLIP from openai.

train on grit-1m

|     model     |     Acc1     |     Acc5     |                      google drive link                       |                        openxlab link                         |
| :-----------: | :----------: | :----------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
|   CLIP-B/16   | 68.31(+1.83) | 90.31(+1.41) | [clip_b16_grit1m_fultune_8xe](https://drive.google.com/file/d/16fHEXZ-7bgzcSBHzEz1wXRIZTjQIXm_2/view?usp=drive_link) | [clip_b16_grit1m_fultune_8xe](https://download.openxlab.org.cn/models/SunzeY/AlphaCLIP/weight/clip_b16_grit1m_fultune_8xe.pth) |
|   CLIP-L/14   | 77.22(+3.74) | 94.38(+2.78) | [clip_l14_grit1m_fultune_8xe](https://drive.google.com/file/d/1PIhplBnsmSWiJN--TXCCSsiaV6bY9koq/view?usp=drive_link) | [clip_l14_grit1m_fultune_8xe](https://download.openxlab.org.cn/models/SunzeY/AlphaCLIP/weight/clip_l14_grit1m_fultune_8xe.pth) |
| CLIP-L/14@336 | 78.15(+3.86) | 94.86(+2.89) | [clip_l14@336_grit1m_fultune_8xe](https://drive.google.com/file/d/1DeNbUv0lraDxJZItb7shTlvGW6z_Z9Si/view?usp=drive_link) | [clip_l14@336_grit1m_fultune_8xe](https://download.openxlab.org.cn/models/SunzeY/AlphaCLIP/weight/clip_l14_336_grit1m_fultune_8xe.pth) |

train on grit-20m

|   model   |     Acc1     |     Acc5     |                      google drive link                       |                        openxlab link                         |
| :-------: | :----------: | :----------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| CLIP-B/16 | 68.89(+2.41) | 90.51(+1.61) | [clip_b16_grit20m_fultune_2xe](https://drive.google.com/file/d/1cj3cYwrzBivx0h0NzSjlCg9HAd5aTkDW/view?usp=sharing) | [clip_b16_grit20m_fultune_2xe](https://download.openxlab.org.cn/models/SunzeY/AlphaCLIP/weight/clip_b16_grit20m_fultune_2xe.pth) |
| CLIP-L/14 | 77.41(+3.93) | 94.45(+2.82) | [clip_l14_grit20m_fultune_2xe](https://drive.google.com/file/d/1WykuBYWePriCVeW5lOwBsgxgeBMzb1nd/view?usp=share_link) | [clip_l14_grit20m_fultune_2xe](https://download.openxlab.org.cn/models/SunzeY/AlphaCLIP/weight/clip_l14_grit20m_fultune_2xe.pth) |
| CLIP-L/14@336 | 79.61(+5.32) | 95.31(+3.34) | [clip_l14@336_grit20m_fultune_4xe](https://drive.google.com/file/d/1dUq1deeLcou26RuxZbBG57m2ALPWev6-/view?usp=drive_link) | [clip_l14_336_grit20m_fultune_4xe](https://download.openxlab.org.cn/models/SunzeY/AlphaCLIP/weight/clip_l14_336_grit_20m_4xe.pth) |

train on combined dataset(mimagenet_top+grit-1m)

|   model   | Imagenet-S Acc1 | Imagenet-S Acc5 | COCO crop Acc1 |                      google drive link                       |                        openxlab link                         |
| :-------: | :-------------: | :-------------: | :------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| CLIP-B/16 |  69.40(+2.92)   |  90.74(+1.84)   |  55.39(+4.97)  | [clip_b16_grit1m+mim_fultune_4xe](https://drive.google.com/file/d/11iDlSAYI_BAi1A_Qz6LTWYHNgPe-UY7I/view?usp=sharing) | [clip_b16_grit1m+mim_fultune_4xe](https://download.openxlab.org.cn/models/SunzeY/AlphaCLIP/weight/clip_b16_grit+mim_fultune_4xe.pth) |
| CLIP-L/14 |  77.80(+4.32)   |  94.46(+2.86)   |  58.83(+3.40)  | [clip_l14_grit1m+mim_fultune_6xe](https://drive.google.com/file/d/1JfzOTvjf0tqBtKWwpBJtjYxdHi-06dbk/view?usp=sharing) | [clip_l14_grit1m+mim_fultune_6xe](https://download.openxlab.org.cn/models/SunzeY/AlphaCLIP/weight/clip_l14_grit+mim_fultune_6xe.pth) |

## Open-CLIP

We are planning to train based on [open_clip](https://github.com/mlfoundations/open_clip) with deeper ViT encoder. please **STAY TUNED**!
