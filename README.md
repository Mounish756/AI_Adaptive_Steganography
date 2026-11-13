# Development of an AI-Based Adaptive Image Embedding Framework for Secure Information Concealment

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)](https://pytorch.org/)
[![Phase](https://img.shields.io/badge/Phase-1%3A%20Foundational%20Research%20%26%20Pipeline-green.svg)]()

> **Academic Project Note**: Phase 1 establishes the foundational Python research infrastructure, AES-256-GCM security pipeline, payload serialization, adaptive feature extraction, baseline benchmark suite, PyTorch neural network scaffolding, unit tests, and research documentation. Phase 1 does **NOT** contain the final trained AI encoder/decoder weights.

---

## Table of Contents

1. [Project Title & Overview](#project-title--overview)
2. [Problem Statement](#problem-statement)
3. [Motivation](#motivation)
4. [Project Objectives](#project-objectives)
5. [Proposed Architecture](#proposed-architecture)
6. [Adaptive Embedding Concept](#adaptive-embedding-concept)
7. [Role of AES-256-GCM Encryption](#role-of-aes-256-gcm-encryption)
8. [Role of Traditional LSB Baseline](#role-of-traditional-lsb-baseline)
9. [Dataset Preparation](#dataset-preparation)
10. [Installation & Setup](#installation--setup)
11. [Running Image Preprocessing](#running-image-preprocessing)
12. [Running Adaptive Map Generation](#running-adaptive-map-generation)
13. [Running LSB Baseline Experiment](#running-lsb-baseline-experiment)
14. [Running Evaluation Metrics & Tests](#running-evaluation-metrics--tests)
15. [Project Structure](#project-structure)
16. [Current Limitations](#current-limitations)
17. [Future AI Model Stages (Phase 2)](#future-ai-model-stages-phase-2)

---

## 1. Project Title & Overview

**Title**: *Development of an AI-Based Adaptive Image Embedding Framework for Secure Information Concealment*

This major academic research project develops a content-aware, deep-learning-guided image steganography system. The core innovation lies in **Content-Aware Adaptive Embedding**: rather than embedding secret payload bits uniformly across pixel locations (which introduces detectable artifacts in smooth regions), the system dynamically analyzes cover image characteristics (gradient edges, local variance, and texture complexity) to generate a spatial embedding suitability map. This suitability map guides a U-Net encoder neural network with spatial/channel attention (CBAM) to conceal encrypted payloads strictly within visually complex image regions that mask distortion.

---

## 2. Problem Statement

Traditional image steganography methods (such as fixed Least Significant Bit substitution or global transform-domain methods) apply embedding changes uniformly across an image. Consequently:
- **Visual Artifacts**: Embedding data into smooth, low-variance regions (such as clear sky or uniform backgrounds) alters pixel statistics in ways human eyes and statistical steganalysis tools can readily detect.
- **Vulnerability to Stegananalysis**: Statistical attacks (e.g. RS steganalysis, Chi-square tests, and sample-pair analysis) easily detect uniform LSB modifications.
- **Lack of Adaptability**: Static embedding algorithms cannot dynamically adjust their payload placement or local embedding intensity based on semantic content or texture variations.

---

## 3. Motivation

With the rapid expansion of digital communications, ensuring data privacy and covert transmission is vital. Encryption protects data content, but encrypted data stream presence is obvious to eavesdroppers. Steganography conceals the *very existence* of the communication. Combining **cryptographic security (AES-256-GCM)** with **content-aware AI steganography** yields a hybrid defense-in-depth framework: even if an adversary suspects secret data transmission, the payload remains cryptographically unreadable without the secret key.

---

## 4. Project Objectives

1. **Secure Payload Pipeline**: Implement AES-256-GCM authenticated encryption with PBKDF2 key derivation and binary payload serialization.
2. **Adaptive Image Analysis**: Construct handcrafted feature maps (Sobel edge, local variance, texture complexity) fused into a normalized spatial suitability map ($256 \times 256 \times 1$) in $[0, 1]$.
3. **Neural Architecture Scaffolding**: Build PyTorch U-Net Encoder, U-Net Decoder, and Convolutional Block Attention Modules (CBAM).
4. **Empirical Baseline Benchmark**: Implement traditional spatial LSB steganography to establish benchmark metrics (MSE, PSNR, SSIM, bpp, BER, latency).
5. **Comprehensive Evaluation Suite**: Build modular metric functions and unit tests verifying security, data loading, preprocessing, metrics, and adaptive map generation.

---

## 5. Proposed Architecture

```
SECRET INFORMATION
       │
       ▼
[AES-256-GCM Encryption] (PBKDF2 HMAC SHA-256)
       │
       ▼
[Encrypted Binary Payload Serialization]
       │
       ▼
┌───────────────────────────────────────────────────────────┐
│                 ADAPTIVE AI SYSTEM                        │
│                                                           │
│  Cover Image                                              │
│       ↓                                                   │
│  [CNN & Edge/Texture/Variance Feature Extraction]        │
│       ↓                                                   │
│  [CBAM Attention Module]                                  │
│       ↓                                                   │
│  [Adaptive Embedding Strength Map (256 × 256 × 1)]        │
│       ↓                                                   │
│  [U-Net Neural Encoder]                                   │
│       ↓                                                   │
│  Stego Image                                              │
└────────────────────────────┬──────────────────────────────┘
                             │
                             ▼
                    [U-Net Neural Decoder]
                             │
                             ▼
                 Recovered Encrypted Payload
                             │
                             ▼
                 [AES-256-GCM Decryption]
                             │
                             ▼
                  Original Secret Information
```

---

## 6. Adaptive Embedding Concept

The central research hypothesis is **Content-Aware Adaptive Embedding**:
- **Smooth Image Regions** (low edge magnitude, low variance): Have low masking capacity. The adaptive analyzer assigns low suitability values ($\to 0.0$), restricting embedding perturbation.
- **Complex / Textured Regions** (high gradient edges, high local variance, intricate texture): Have high human visual system (HVS) masking capacity. The adaptive analyzer assigns high suitability values ($\to 1.0$), concentrating payload embedding perturbations where they are imperceptible.

The fused handcrafted suitability map $S(x, y)$ is defined as:
$$S(x, y) = \text{Normalize}\left( w_e \cdot E(x, y) + w_v \cdot V(x, y) + w_t \cdot T(x, y) \right)$$
where $E(x,y)$ is Sobel edge magnitude, $V(x,y)$ is local variance, and $T(x,y)$ is Laplacian texture response.

---

## 7. Role of AES-256-GCM Encryption

Encryption is integrated prior to steganographic embedding to ensure multi-layered security:
- **Confidentiality & Authenticity**: AES-GCM is an Authenticated Encryption with Associated Data (AEAD) cipher. It protects against message recovery and payload tampering via a 128-bit authentication tag.
- **Key Derivation**: Passwords are stretched using PBKDF2-HMAC-SHA256 with a unique 16-byte cryptographic salt.
- **Zero Hardcoding**: Encryption keys and passwords are loaded dynamically or passed at runtime—never stored in source code.

---

## 8. Role of Traditional LSB Baseline

Traditional 1-bit Least Significant Bit (LSB) steganography is included **strictly as an empirical baseline benchmark**.
- It provides a reference baseline for measuring PSNR, SSIM, MSE, and execution latency.
- It is explicitly labeled `"Traditional LSB Baseline"` in code and logs to distinguish it from the proposed AI model.

---

## 9. Dataset Preparation

Place cover images in the dataset split directories:

```
AI-Adaptive-Steganography/datasets/
├── train/       # Place 70% of cover images here
├── validation/  # Place 15% of cover images here
└── test/        # Place 15% of cover images here
```

- **Supported Formats**: PNG, JPG, JPEG, BMP.
- **Recommended Public Datasets**: BOSSbase 1.01, COCO 2017, ImageNet subset, or custom natural images.
- **Automatic Synthetic Generation**: If no images are placed in the dataset directories, running the provided script `python scripts/generate_synthetic_data.py` populates sample images automatically for testing.

---

## 10. Installation & Setup

### Prerequisites
- Python 3.10+
- PyTorch 2.0+

### Step-by-Step Installation

```bash
# 1. Navigate to project root
cd AI-Adaptive-Steganography

# 2. Create virtual environment (optional but recommended)
python -m venv venv
# On Windows PowerShell:
.\venv\Scripts\Activate.ps1

# 3. Install dependencies
pip install -r requirements.txt
```

---

## 11. Running Image Preprocessing

To generate synthetic cover images and test preprocessing pipeline functionality:

```bash
# Generate synthetic dataset samples
python scripts/generate_synthetic_data.py
```

---

## 12. Running Adaptive Map Generation

To run the handcrafted adaptive feature analyzer and save visualization grids to `results/adaptive_maps/`:

```bash
python scripts/visualize_adaptive_map.py
```

Generated plots illustrate the Original Image, Sobel Edge Map, Local Variance Map, Texture Complexity Map, and Final Adaptive Suitability Map.

---

## 13. Running LSB Baseline Experiment

To run an end-to-end baseline experiment (AES-GCM encryption $\to$ LSB embedding $\to$ LSB extraction $\to$ AES-GCM decryption $\to$ evaluation metrics):

```bash
python scripts/run_lsb_experiment.py
```

---

## 14. Running Evaluation Metrics & Tests

Execute the complete PyTest unit test suite:

```bash
pytest tests/ -v
```

Tests cover Preprocessing, Dataset Partitioning, AES-256-GCM Security, Payload Serialization, Baseline LSB, Evaluation Metrics, and Adaptive Map Generation.

---

## 15. Project Structure

```
AI-Adaptive-Steganography/
├── ai/
│   ├── models/
│   │   ├── adaptive_analyzer.py # Handcrafted & Neural adaptive suitability map modules
│   │   ├── encoder.py           # PyTorch U-Net Encoder module
│   │   ├── decoder.py           # PyTorch U-Net Decoder module
│   │   ├── attention.py         # CBAM Channel & Spatial Attention module
│   │   └── full_model.py        # End-to-end AdaptiveStegoModel container
│   ├── training/
│   │   ├── train.py             # Training loop scaffold
│   │   ├── validate.py          # Validation loop module
│   │   ├── test.py              # Test evaluation module
│   │   └── losses.py            # Multi-objective steganography loss function
│   ├── data/
│   │   ├── dataset.py           # PyTorch SteganographyDataset (70/15/15 split)
│   │   └── preprocessing.py     # Resizing, RGB conversion, normalization [0, 1]
│   └── metrics/
│       ├── psnr.py              # Peak Signal-to-Noise Ratio (dB)
│       ├── ssim.py              # Structural Similarity Index Measure
│       ├── mse.py               # Mean Squared Error
│       └── payload.py           # bpp, BER, latency metrics
├── security/
│   ├── encryption.py            # AES-256-GCM encryption & decryption
│   ├── key_management.py        # Key derivation & environment key loading
│   └── payload.py               # Payload serialization & bitstream conversions
├── baselines/
│   └── lsb.py                   # Traditional 1-bit LSB steganography baseline
├── experiments/
│   └── README.md                # Research experiment log
├── datasets/
│   ├── train/                   # Training dataset split
│   ├── validation/              # Validation dataset split
│   └── test/                    # Test dataset split
├── models/                      # Saved PyTorch model checkpoints
├── results/
│   └── adaptive_maps/           # Output visualization plots
├── scripts/
│   ├── generate_synthetic_data.py
│   ├── visualize_adaptive_map.py
│   └── run_lsb_experiment.py
├── tests/                       # Complete PyTest test suite
├── configs/
│   └── config.yaml              # Centralized configuration parameters
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 16. Current Limitations

1. **Phase 1 Scope**: Phase 1 provides full functional code, algorithms, baseline experiments, and neural network module scaffolding, but does not contain trained neural network weights.
2. **Handcrafted Feature Heuristic**: The initial suitability map relies on weighted linear fusion of handcrafted OpenCV features, which serves as the reference input for Phase 2 neural network learning.
3. **Payload Compression**: Input payload bytes are encrypted and serialized; explicit entropy compression (e.g. Huffman or LZMA) before encryption can be explored in Phase 2.

---

## 17. Future AI Model Stages (Phase 2)

- **Phase 2A**: Train the U-Net Encoder and Decoder on natural image datasets (BOSSbase / COCO) using the loss functions defined in `ai/training/losses.py`.
- **Phase 2B**: Fine-tune the Neural Adaptive Analyzer to jointly optimize visual quality (PSNR > 45 dB, SSIM > 0.98) and payload capacity.
- **Phase 2C**: Evaluate robustness against steganalysis tools (Aletheia, StegExpose) and common image degradation attacks (JPEG compression, Gaussian noise, cropping).
