# Technical Architecture Documentation: AI-Based Adaptive Region Analyzer

## 1. Problem Statement

Traditional image steganography methods (such as fixed spatial LSB substitution or uniform transform-domain embedding) apply payload embedding operations uniformly across all image pixels regardless of local visual content. 
- **Smooth Image Regions**: Flat, low-variance image regions (e.g., clear sky, uniform backgrounds) possess minimal human visual system (HVS) noise masking capacity. Embedding data into these areas alters local statistical distributions, introducing visually perceptible artifacts and making the stego image highly vulnerable to statistical steganalysis attacks (such as RS steganalysis and Chi-square tests).
- **Static Embedding Rules**: Traditional heuristics (e.g. fixed Canny/Sobel edge detectors) rely on hard-coded threshold rules, lacking the ability to learn complex spatial feature patterns or adapt dynamically during neural network training.

---

## 2. Proposed Solution

The **AI-Based Adaptive Region Analyzer** introduces a trainable PyTorch deep convolutional neural network (`AdaptiveRegionAnalyzer`) that processes an input cover image and outputs a continuous, spatial **Adaptive Embedding Suitability Map** $S(x, y) \in [0.0, 1.0]$.
- **High Suitability ($S(x, y) \to 1.0$)**: Identifies complex textures, high-frequency edge boundaries, and high local variance regions capable of masking embedding distortion.
- **Low Suitability ($S(x, y) \to 0.0$)**: Identifies smooth, uniform regions where embedding perturbations are restricted.

---

## 3. Network Architecture

```
                       INPUT COVER IMAGE (B, 3, 256, 256)
                                      │
                                      ▼
                       [Initial Stem Block (3 -> 32)]
                                      │
                                      ▼
                        [Encoder Stage 1 (32 -> 64)]
                                      │
                                      ▼
                       [CBAM Attention Block (64)]
                                      │
                                      ▼
                        [Encoder Stage 2 (64 -> 128)]
                                      │
                                      ▼
                       [CBAM Attention Block (128)]
                                      │
                                      ▼
                        [Bottleneck Block (128 -> 128)]
                                      │
                                      ▼
                 [Decoder Stage 1 (ConvTranspose2d 128 -> 64)]
                                      │
                                      ▼
                 [Decoder Stage 2 (ConvTranspose2d 64 -> 32)]
                                      │
                                      ▼
                      [Final Output Conv 1x1 (32 -> 1)]
                                      │
                                      ▼
                       [Sigmoid Activation Function]
                                      │
                                      ▼
               ADAPTIVE SUITABILITY MAP S(x, y) ∈ [0.0, 1.0]
                          Shape: (B, 1, 256, 256)
```

### Component Details
1. **Multi-Scale Convolutional Feature Extraction**: Extracts low-level edge features and high-level semantic texture representations across multiple spatial scales.
2. **CBAM Attention Modules**: Combines Channel Attention (recalibrating feature channels) and Spatial Attention (highlighting spatial regions) to focus model capacity on feature-dense regions.
3. **Residual Skip Connections**: Connects encoder features to upsampling decoder stages to preserve fine-grained spatial boundaries.
4. **Sigmoid Activation**: Constrains output suitability values strictly within the continuous range $[0.0, 1.0]$.

---

## 4. Input & Output Specifications

- **Input Tensor**:
  $$\mathbf{X} \in \mathbb{R}^{B \times 3 \times 256 \times 256}, \quad X_{i, j, k} \in [0.0, 1.0] \text{ (float32 RGB)}$$

- **Output Tensor**:
  $$\mathbf{S} \in \mathbb{R}^{B \times 1 \times 256 \times 256}, \quad S_{i, j, k} \in [0.0, 1.0] \text{ (float32 Suitability Map)}$$

---

## 5. Training Strategy & Loss Formulation

The network is trained using a reference suitability map $S_{\text{ref}}$ constructed from a weighted combination of Sobel edge magnitude, local variance (7x7 window), and Laplacian texture response minus a smooth-region penalty:
$$S_{\text{ref}}(x, y) = \text{Normalize}\left( w_e \cdot E(x, y) + w_v \cdot V(x, y) + w_t \cdot T(x, y) - w_s \cdot P_{\text{smooth}}(x, y) \right)$$

### Objective Function
$$\mathcal{L}_{\text{total}} = \lambda_{\text{rec}} \cdot \text{MSE}(\hat{S}, S_{\text{ref}}) + \lambda_{\text{smooth}} \cdot \text{TV}(\hat{S})$$
where $\text{TV}(\hat{S})$ is Total Variation loss promoting spatial smoothness and preventing pixel-wise noise artifacts.

---

## 6. Current Limitations & Scope

> [!WARNING]
> **Scientific Limitation Statement**: The Adaptive Region Analyzer generates a spatial suitability map estimating visual distortion masking capacity. Generating an adaptive suitability map alone does **not** prove end-to-end steganographic security or un-detectability. End-to-end security requires joint training with the neural Encoder and Decoder in subsequent project phases.
