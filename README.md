# DCT-Mask: Discrete Cosine Transform Mask Representation for Instance Segmentation

This project hosts the code for implementing the DCT-MASK algorithms for instance segmentation.

> [**DCT-Mask: Discrete Cosine Transform Mask Representation for Instance Segmentation**]
> Xing Shen*, Jirui Yang*, Chunbo Wei, Bing Deng, Jianqiang Huang, Xiansheng Hua
 Xiaoliang Cheng, Kewei Liang
>
> In Proceedings of the IEEE conference on Computer Vision and Pattern Recognition(CVPR 2021)
>
> *arXiv preprint([arXiv:2011.09876](https://arxiv.org/abs/2011.09876))*  

## Contributions
- We propose a high-quality and low-complexity mask representation for instance segmentation, which encodes the high-resolution binary mask into a compact vector with discrete cosine transform.
- With slight modifications, DCT-Mask could be integrated
  into most pixel-based frameworks, and achieve
  significant and consistent improvement on different
  datasets, backbones, and training schedules. Specifically,
  it obtains more improvements for more complex
  backbones and higher-quality annotations.
- DCT-Mask does not require extra pre-processing or
  pre-training. It achieves high-resolution mask prediction
  at a speed similar to low-resolution.

## Installation
#### Requirements
- PyTorch ≥ 1.5 and fvcore == 0.1.1.post20200716

This implementation is based on [detectron2](https://github.com/facebookresearch/detectron2). Please refer to [INSTALL.md](INSTALL.md). for installation and dataset preparation.

## Usage 
The codes of this project is on projects/DCT_Mask/ 
### Train with multiple GPUs
    cd ./projects/DCT_Mask/
    ./train1.sh

### Testing
    cd ./projects/DCT_Mask/
    ./test1.sh
## Model ZOO 
### Trained models on COCO
Model |  Backbone | Schedule | Multi-scale training | Inference time (s/im) | AP (minival) | Link
--- |:---:|:---:|:---:|:---:|:---:|:---:
DCT-Mask R-CNN | R50 | 1x | Yes |   0.0465 | 36.5  | [download(Fetch code: xpdm)](https://pan.baidu.com/s/1p1OK9KU3ojVwM0gj8nqkPw)
DCT-Mask R-CNN | R101 | 3x | Yes |   0.0595 | 39.9  | [download(Fetch code: 7q6x)](https://pan.baidu.com/s/19IYgrUXi4o_gTNl8MzGIOA)
DCT-Mask R-CNN | RX101 | 3x | Yes |   0.1049 | 41.2  | [download(Fetch code: ufw2)](https://pan.baidu.com/s/149NL1S4AfJJSRSki3bVpGw)
Casecade DCT-Mask R-CNN | R50  | 1x | Yes |   0.0630 | 37.5  | [download(Fetch code: yqxp)](https://pan.baidu.com/s/1U9AF8bP5FTWYqBGVrt5HmA)
Casecade DCT-Mask R-CNN | R101  | 3x | Yes |   0.0750 | 40.8  | [download(Fetch code: r8xv)](https://pan.baidu.com/s/11UQ1Zot7M5FqK1DIa-HOHA)
Casecade DCT-Mask R-CNN | RX101  | 3x | Yes |   0.1195 | 42.0  | [download(Fetch code: pdej)](https://pan.baidu.com/s/1xaChv_C-YRxkxY6gjumHOw)

### Trained models on Cityscapes
Model |Data|  Backbone | Schedule | Multi-scale training | AP (val) | Link
--- |:---:|:---:|:---:|:---:|:---:|:---:
DCT-Mask R-CNN | Fine-Only | R50 | 1x | Yes | 37.0  | [download(Fetch code: dn7i)](https://pan.baidu.com/s/1vcDVv8NbOm3OV8_2fsf-DQ)
DCT-Mask R-CNN | CoCo-Pretrain +Fine | R50 | 1x | Yes | 39.6  | [download(Fetch code: ntqf)](https://pan.baidu.com/s/1dVcSwP2PG_6jZYgVMbWT0w)

#### Notes
- We observe about 0.2 AP noise in COCO.
- High variance observed in CityScapes when trained on fine annotations only. 
  We report the median of 5 runs AP in the article (i.e. 35.6), while in this repo we report the best results (37.0).
- Initialized from COCO pre-training will reduce the variance on CityScapes as well as increasing mask AP.
- The inference time is measured on single GPU with batchsize 1. All GPUs are NVIDIA V100.
- [Lvis 0.5](https://) is used for evaluation.

## Contributing to the project
Any pull requests or issues are welcome. 

If there is any problem with this project, please contact [Xing Shen](shenxingsx@zju.edu.cn).

## Citations
Please consider citing our papers in your publications if the project helps your research. 

## License
- MIT License.