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
