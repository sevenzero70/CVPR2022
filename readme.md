# 🍎 low-light

## 1 Toward Fast, Flexible, and Robust Low-Light Image Enhancement
[Paper](http://arxiv.org/pdf/2204.10137) /
[Code](https://github.com/vis-opt-group/SCI) <br>
> L. Ma, T. Ma, R. Liu, X. Fan, and Z. Luo, “Toward Fast, Flexible, and Robust Low-Light Image Enhancement,” arXiv:2204.10137 [cs], Apr. 2022, Accessed: May 06, 2022. [Online]. Available: http://arxiv.org/abs/2204.10137

Existing low-light image enhancement techniques are mostly not only difficult to deal with both visual quality and computational efficiency but also commonly invalid in unknown complex scenarios. In this paper, we develop a new Self-Calibrated Illumination (SCI) learning framework for fast, flexible, and robust brightening images in real-world low-light scenarios. To be specific, we establish a cascaded illumination learning process with weight sharing to handle this task. Considering the computational burden of the cascaded pattern, we construct the self-calibrated module which realizes the convergence between results of each stage, producing the gains that only use the single basic block for inference (yet has not been exploited in previous works), which drastically diminishes computation cost. We then define the unsupervised training loss to elevate the model capability that can adapt to general scenes. Further, we make comprehensive explorations to excavate SCI's inherent properties (lacking in existing works) including operation-insensitive adaptability (acquiring stable performance under the settings of different simple operations) and model-irrelevant generality (can be applied to illumination-based existing works to improve performance). Finally, plenty of experiments and ablation studies fully indicate our superiority in both quality and efficiency. Applications on low-light face detection and nighttime semantic segmentation fully reveal the latent practical values for SCI. The source code is available at https://github.com/vis-opt-group/SCI.

![image](https://user-images.githubusercontent.com/70806159/167101488-fadd8ff3-e502-4027-8e1b-d185b9ca3b26.png)

------
# Dehazing/Denoising

## 1 Image dehazing transformer with transmission-aware 3D position embedding
[paper](https://li-chongyi.github.io/Proj_DeHamer.html)
> “Image Dehazing Transformer with Transmission-Aware 3D Position Embedding.” https://li-chongyi.github.io/Proj_DeHamer.html (accessed May 06, 2022).

Despite single image dehazing has been made promising progress with Convolutional Neural Networks (CNNs), the inherent equivariance and locality of convolution still bottleneck dehazing performance. Though Transformer has occupied various computer vision tasks, directly leveraging Transformer for image dehazing is challenging: 1) it tends to result in ambiguous and coarse details that are undesired for image reconstruction; 2) previous position embedding of Transformer is provided in logic or spatial position order that neglects the variational haze densities, which results in the sub-optimal dehazing performance. **The key insight of this study is to investigate how to combine CNN and Transformer for image dehazing.** To solve the feature inconsistency issue between Transformer and CNN, we propose to modulate CNN features via learning modulation matrices (i.e., coefficient matrix and bias matrix) conditioned on Transformer features instead of simple feature addition or concatenation. The feature modulation naturally inherits the global context modeling capability of Transformer and the local representation capability of CNN. We bring a haze density-related prior into Transformer via a novel transmission-aware 3D position embedding module, which not only provides the relative position but also suggests the haze density of different spatial regions. Extensive experiments demonstrate that our method attains state-of-the-art performance on several image dehazing benchmarks.

![image](https://user-images.githubusercontent.com/70806159/167100852-a42ff08f-dc51-4a1e-9f1f-f621e4af67dc.png)
