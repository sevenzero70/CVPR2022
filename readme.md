# 🍎 low-light

## 1 Toward Fast, Flexible, and Robust Low-Light Image Enhancement
[Paper](http://arxiv.org/pdf/2204.10137) /
[Code](https://github.com/vis-opt-group/SCI) <br>
> L. Ma, T. Ma, R. Liu, X. Fan, and Z. Luo, “Toward Fast, Flexible, and Robust Low-Light Image Enhancement,” arXiv:2204.10137 [cs], Apr. 2022, Accessed: May 06, 2022. [Online]. Available: http://arxiv.org/abs/2204.10137

Existing low-light image enhancement techniques are mostly not only difficult to deal with both visual quality and computational efficiency but also commonly invalid in unknown complex scenarios. In this paper, we develop a new Self-Calibrated Illumination (SCI) learning framework for fast, flexible, and robust brightening images in real-world low-light scenarios. To be specific, we establish a cascaded illumination learning process with weight sharing to handle this task. Considering the computational burden of the cascaded pattern, we construct the self-calibrated module which realizes the convergence between results of each stage, producing the gains that only use the single basic block for inference (yet has not been exploited in previous works), which drastically diminishes computation cost. We then define the unsupervised training loss to elevate the model capability that can adapt to general scenes. Further, we make comprehensive explorations to excavate SCI's inherent properties (lacking in existing works) including operation-insensitive adaptability (acquiring stable performance under the settings of different simple operations) and model-irrelevant generality (can be applied to illumination-based existing works to improve performance). Finally, plenty of experiments and ablation studies fully indicate our superiority in both quality and efficiency. Applications on low-light face detection and nighttime semantic segmentation fully reveal the latent practical values for SCI. The source code is available at https://github.com/vis-opt-group/SCI.

![image](https://user-images.githubusercontent.com/70806159/167101488-fadd8ff3-e502-4027-8e1b-d185b9ca3b26.png)

------
# Dehazing/Denoising

## 1. Image dehazing transformer with transmission-aware 3D position embedding
[paper](https://li-chongyi.github.io/Proj_DeHamer.html)
> “Image Dehazing Transformer with Transmission-Aware 3D Position Embedding.” https://li-chongyi.github.io/Proj_DeHamer.html (accessed May 06, 2022).

Despite single image dehazing has been made promising progress with Convolutional Neural Networks (CNNs), the inherent equivariance and locality of convolution still bottleneck dehazing performance. Though Transformer has occupied various computer vision tasks, directly leveraging Transformer for image dehazing is challenging: 1) it tends to result in ambiguous and coarse details that are undesired for image reconstruction; 2) previous position embedding of Transformer is provided in logic or spatial position order that neglects the variational haze densities, which results in the sub-optimal dehazing performance. **The key insight of this study is to investigate how to combine CNN and Transformer for image dehazing.** To solve the feature inconsistency issue between Transformer and CNN, we propose to modulate CNN features via learning modulation matrices (i.e., coefficient matrix and bias matrix) conditioned on Transformer features instead of simple feature addition or concatenation. The feature modulation naturally inherits the global context modeling capability of Transformer and the local representation capability of CNN. We bring a haze density-related prior into Transformer via a novel transmission-aware 3D position embedding module, which not only provides the relative position but also suggests the haze density of different spatial regions. Extensive experiments demonstrate that our method attains state-of-the-art performance on several image dehazing benchmarks.

![image](https://user-images.githubusercontent.com/70806159/167100852-a42ff08f-dc51-4a1e-9f1f-f621e4af67dc.png)

## 2. **Blind2Unblind: Self-Supervised Image Denoising with Visible Blind Spots**

[Paper](https://arxiv.org/abs/2203.06967)/
[Code](https://github.com/demonsjin/Blind2Unblind)

Real noisy-clean pairs on a large scale are costly and difficult to obtain. Meanwhile, supervised denoisers trained on synthetic data perform poorly in practice. Self- supervised denoisers, which learn only from single noisy images, solve the data collection problem. However, self- supervised denoising methods, especially blindspot-driven ones, suffer sizable information loss during input or net- work design. The absence of valuable information dramat- ically reduces the upper bound of denoising performance. In this paper, we propose a simple yet efficient approach called Blind2Unblind to overcome the information loss in blindspot-driven denoising methods. First, we introduce a global-aware mask mapper that enables global perception and accelerates training. The mask mapper samples all pix- els at blind spots on denoised volumes and maps them to the same channel, allowing the loss function to optimize all blind spots at once. Second, we propose a re-visible loss to train the denoising network and make blind spots visible. The denoiser can learn directly from raw noise im- ages without losing information or being trapped in identity mapping. We also theoretically analyze the convergence of the re-visible loss. Extensive experiments on synthetic and real-world datasets demonstrate the superior performance of our approach compared to previous work. Code is avail- able at https://github.com/demonsjin/Blind2Unblind.

![image](https://user-images.githubusercontent.com/70806159/167102522-fc241c65-333e-45e2-b25a-f873829ca2cb.png)

## 3. **CVF-SID: Cyclic multi-Variate Function for Self-Supervised Image Denoising by Disentangling Noise from Image**

[paper](https://arxiv.org/pdf/2203.13009.pdf)

Recently, significant progress has been made on image
denoising with strong supervision from large-scale datasets.
However, obtaining well-aligned noisy-clean training image pairs for each specific scenario is complicated and
costly in practice. Consequently, applying a conventional
supervised denoising network on in-the-wild noisy inputs is
not straightforward. Although several studies have challenged this problem without strong supervision, they rely on
less practical assumptions and cannot be applied to practical situations directly. To address the aforementioned challenges, we propose a novel and powerful self-supervised denoising method called CVF-SID based on a Cyclic multiVariate Function (CVF) module and a self-supervised image disentangling (SID) framework. The CVF module can
output multiple decomposed variables of the input and take
a combination of the outputs back as an input in a cyclic
manner. Our CVF-SID can disentangle a clean image
and noise maps from the input by leveraging various selfsupervised loss terms. Unlike several methods that only
consider the signal-independent noise models, we also deal
with signal-dependent noise components for real-world applications. Furthermore, we do not rely on any prior assumptions about the underlying noise distribution, making
CVF-SID more generalizable toward realistic noise. Extensive experiments on real-world datasets show that CVFSID achieves state-of-the-art self-supervised image denoising performance and is comparable to other existing approaches. The code is publicly available from this link.
