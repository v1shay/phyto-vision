<div align="center">

<h1><img width="1086" height="362" alt="ChatGPT Image Aug 12, 2026, 01_51_33 AM" src="https://github.com/user-attachments/assets/6ad88c9f-247c-43aa-a0e1-af0a73388245" /></h1>

<p><strong>Computer vision-powered deep learning for plant health diagnosis.</strong></p>

</div>

---

## Results

- **Accuracy:** 95%+ validation accuracy  
- **Dataset:** 395K+ plant images (Kaggle, PlantVillage) 
- **Classes:** 5 plant disease categories  
- **Model:** Custom CNN image-classification architecture  
- **Deployment:** Presented to the UC Santa Cruz Environmental Science Department for potential greenhouse integration  

---

## Overview

PhytoVision is a computer vision system for detecting plant disease from leaf imagery. The system uses a convolutional neural network to classify plant health conditions from high-resolution image inputs, with emphasis on real-world greenhouse monitoring.

The pipeline converts raw plant images into normalized tensors, applies supervised deep learning, and outputs disease classifications that can support early diagnosis and crop-loss prevention.

---

## Method / Approach

<p align="center">
  <img width="1059" height="635" alt="PhytoVision Dashboard Screenshot" src="https://github.com/user-attachments/assets/65195b0e-da3f-46dc-be85-016a568bd563" />
</p>

- **Image Standardization**  
  Plant images are resized, normalized, and prepared for CNN-based inference.

- **Feature Learning**  
  The CNN learns visual disease markers directly from image data, including:
  - leaf discoloration patterns  
  - lesion texture and shape  
  - edge degradation and surface irregularities  

- **Supervised Classification**  
  Models map image features → plant health class:
  - disease classification objective  
  - 5-category output space  

- **Evaluation + Deployment Path**  
  Models are validated on held-out image data and designed for greenhouse monitoring integration.

---

## Data

- **Source:** proprietary / aggregated plant image corpus  
- **Type:** high-resolution plant disease image dataset  
- **Size:** 395K+ images  
- **Classes:** 5 disease categories  

<p align="center">
  <img width="1536" height="1024" alt="PhytoVision Plant Disease Image Grid" src="https://github.com/user-attachments/assets/4bdb993a-533b-4f8d-b462-300c969a59d3" />
</p>

Preprocessing:
- image resizing  
- normalization  
- augmentation  
- train / validation partitioning  

---

## Experiments / Reproduction

```bash
python training/train.py
python evaluation/evaluate.py
````

## Run inference:

```bash
python api/infer.py --input sample_leaf.jpg
```

## Train model:

```bash
python training/train.py --config configs/default.yaml
```

Input: plant image
Output: disease class + confidence score

Dependencies

```bash
Python 3.x
NumPy
PyTorch
TorchVision
Pillow
OpenCV
FastAPI / Flask
```

## Repository Structure

```bash
phytovision/
├── data/
│   ├── raw/
│   └── processed/
├── models/
├── training/
├── api/
├── evaluation/
├── notebooks/
└── README.md
```

## Installation

```bash
git clone https://github.com/v1shay/phyto-vision.git
cd phyto-vision
pip install -r requirements.txt
```

## Optional:

```bash
conda env create -f environment.yml
conda activate phytovision
```

