# DINO-Reg

Official repository for the MICCAI 2024 paper:  
**[DINO-Reg: General Purpose Image Encoder for Training-Free Multi-modal Deformable Medical Image Registration](https://papers.miccai.org/miccai-2024/paper/2230_paper.pdf)**

---

**DINO-Reg** is a simple, powerful, and training-free framework for deformable medical image registration.  
Built on top of the DINOv2 ViT backbone, it enables multi-modal registration with general-purpose image features.

---

## 📁 Repository Structure

- `inference_l2rmrct.py` — Main script for inference.
- `models/` — First run will download DINOv2 model here (`dinov2_vitl14_reg4_pretrain.pth`). 
- `dinov2/` — Contains config files and model architecture.
- `sample_dataset_dir/` — Placeholder directory for input images (user needs to populate this).

---

## ⚙️ Environment Setup

We recommend using the **same environment as [DINOv2](https://github.com/facebookresearch/dinov2)**.  
Follow the DINOv2 setup instructions to install the necessary dependencies.

---

## 🔧 How to Run

1. **Populate your dataset**  
   Add your input image pairs into the `sample_dataset_dir/` directory. Modify the csv files accordingly. 
   
   `pairs_Tr.csv` should contain the filenames, and `structures.csv` contains the index for the organ segmentations that need to be evaluated. 

2. **Run Inference**  
   ```bash
   python inference_l2rmrct.py
