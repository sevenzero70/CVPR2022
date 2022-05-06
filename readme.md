# 🍎 low-light

## Toward Fast, Flexible, and Robust Low-Light Image Enhancement
[Paper](http://arxiv.org/pdf/2204.10137) /
[Code](https://github.com/vis-opt-group/SCI) <br>
> [1]L. Ma, T. Ma, R. Liu, X. Fan, and Z. Luo, “Toward Fast, Flexible, and Robust Low-Light Image Enhancement,” arXiv:2204.10137 [cs], Apr. 2022, Accessed: May 06, 2022. [Online]. Available: http://arxiv.org/abs/2204.10137

Existing low-light image enhancement techniques are mostly not only difficult to deal with both visual quality and computational efficiency but also commonly invalid in unknown complex scenarios. In this paper, we develop a new Self-Calibrated Illumination (SCI) learning framework for fast, flexible, and robust brightening images in real-world low-light scenarios. To be specific, we establish a cascaded illumination learning process with weight sharing to handle this task. Considering the computational burden of the cascaded pattern, we construct the self-calibrated module which realizes the convergence between results of each stage, producing the gains that only use the single basic block for inference (yet has not been exploited in previous works), which drastically diminishes computation cost. We then define the unsupervised training loss to elevate the model capability that can adapt to general scenes. Further, we make comprehensive explorations to excavate SCI's inherent properties (lacking in existing works) including operation-insensitive adaptability (acquiring stable performance under the settings of different simple operations) and model-irrelevant generality (can be applied to illumination-based existing works to improve performance). Finally, plenty of experiments and ablation studies fully indicate our superiority in both quality and efficiency. Applications on low-light face detection and nighttime semantic segmentation fully reveal the latent practical values for SCI. The source code is available at https://github.com/vis-opt-group/SCI.

# Dehazing/Denoising

## **Image dehazing transformer with transmission-aware 3D position embedding**
