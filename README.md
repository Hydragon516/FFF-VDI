<div align="center">

<h3> Video Diffusion Models are Strong Video Inpainter
 </h3> 
 <br/>
  <a href='https://arxiv.org/abs/2408.11402'><img src='https://img.shields.io/badge/ArXiv-2303.08314-red' /></a> 
  <br/>
  <br/>
<div>
    <a href='https://hydragon.co.kr' target='_blank'>Minhyeok Lee <sup> 1</sup> </a>&emsp;
    <a href='https://suhwan-cho.github.io' target='_blank'>Suhwan Cho <sup> 1</sup></a>&emsp;
    <a target='_blank'>Chajin Shin <sup> 1</sup></a>&emsp;
    <a href='https://jho-yonsei.github.io' target='_blank'>Jungho Lee <sup> 1</sup></a>&emsp;
    <a target='_blank'>Sunghun Yang <sup> 1</sup></a>&emsp;
    <a target='_blank'>Sangyoun Lee <sup>1</sup></a>&emsp;
</div>
<br>
<div>
                      <sup>1</sup> Yonsei University &nbsp;&nbsp;&nbsp;
</div>
<br>
<i><strong><a href='https://aaai.org/conference/aaai/aaai-25/' target='_blank'>AAAI 2025</a></strong></i>
<br>
<br>
</div>

## Abstract
Propagation-based video inpainting using optical flow at the pixel or feature level has recently garnered significant attention. However, it has limitations such as the inaccuracy of optical flow prediction and the propagation of noise over time. These issues result in non-uniform noise and time consistency problems throughout the video, which are particularly pronounced when the removed area is large and involves substantial movement. To address these issues, we propose a novel First Frame Filling Video Diffusion Inpainting model (FFF-VDI). We design FFF-VDI inspired by the capabilities of pre-trained image-to-video diffusion models that can transform the first frame image into a highly natural video. To apply this to the video inpainting task, we propagate the noise latent information of future frames to fill the masked areas of the first frame's noise latent code. Next, we fine-tune the pre-trained image-to-video diffusion model to generate the inpainted video. The proposed model addresses the limitations of existing methods that rely on optical flow quality, producing much more natural and temporally consistent videos. This proposed approach is the first to effectively integrate image-to-video diffusion models into video inpainting tasks. Through various comparative experiments, we demonstrate that the proposed model can robustly handle diverse inpainting types with high quality.

## Overview
<p align="center">
  <img width="100%" alt="teaser" src="./assets/bmx-trees.gif">
</p>

## Installation
To set up the repository locally, follow these steps:
1. Clone the repository and navigate to the project directory:
    ```bash
    git clone https://github.com/Hydragon516/FFF-VDI.git
    cd FFF-VDI
    ```
2. Create a new conda environment and activate it:
   ```bash
    conda create -n fff-vdi python=3.10
    conda activate fff-vdi
    ```
3. Install torch and other dependencies:
   ```bash
   pip install torch torchvision
   pip install -r requirements.txt
   ```
You need [accelerate](https://github.com/huggingface/accelerate) for model training, so you should configure accelerate based on your hardware setup. Use the following command to configure it:
```bash
accelerate config
```

## Datasets
We use the [YouTube-VOS]([http://saliencydetection.net/duts](https://youtube-vos.org/dataset/vos/)) train dataset for model training. Since FFF-VDI generates random masks during training, it only requires a set of RGB images. The complete dataset directory structure is as follows:

```
.
└── dataset root/
    └── youtube-vos/
        └── JPEGImages/
            ├── 00a23ccf53
            ├── 00ad5016a4
            └── ...
```

## TODO
- [ ] Add training details 
- [ ] Add DNA module
- [ ] Add long video inference code
