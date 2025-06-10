# [CVPR 2025] FasterPET (FPET)

The official pytorch implementation of FasterPET (FPET) from our paper [Faster Parameter-Efficient Tuning with Token Redundancy Reduction](https://arxiv.org/abs/2503.20282)

Kwonyoung Kim<sup>1</sup> Jungin Park<sup>1*</sup> Jin Kim<sup>1</sup> Hyeongjun Kwon<sup>1</sup> Kwanghoon Sohn<sup>1*</sup>

<sup>1</sup>Yonsei University  <sup>*</sup>Corresponding author


## Installation
```
conda create -n fpet python=3.10.13
conda install -r requirements.txt
```

## Data Preparation
To download the datasets, please refer to https://github.com/ZhangYuanhan-AI/NOAH/#data-preparation. Set the directory of the dataset as `<YOUR PATH>/fpet/data/`.

## Pretrained Model
Download the [pretrained ViT-B/16](https://storage.googleapis.com/vit_models/imagenet21k/ViT-B_16.npz) to `<YOUR PATH>/fpet/ViT-B_16.npz`.

## Train & Evaluation
Train and evaluation on VTAB-1K.
```
sh train.sh
```

## Citation
```
@inproceedings{kim2025faster,
  title={Faster Parameter-Efficient Tuning with Token Redundancy Reduction},
  author={Kim, Kwonyoung and Park, Jungin and Kim, Jin and Kwon, Hyeongjun and Sohn, Kwanghoon},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2025}
}
```

## Acknowledgements
Our implementation is based on [NOAH](https://github.com/ZhangYuanhan-AI/NOAH), [timm](https://github.com/rwightman/pytorch-image-models), [Binary Adapter](https://github.com/JieShibo/PETL-ViT/tree/main/binary_adapter) and [ToMe](https://github.com/facebookresearch/tome). Thanks for their awesome works.