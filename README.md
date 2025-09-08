# **DLAS-Net:Dual-Level Anomaly Synthesis Network for Weak Anomaly Detection in Liquid Crystal Displays**

![Fig2_框架图3](https://github.com/user-attachments/assets/d7986cac-8d1e-4297-885b-9faca3d803d8)



## **Introduction**
This repo contains the official PyTorch implementation of DLAS-Net, a dual-level anomaly synthesis network for self-supervised anomaly detection in display, with a particular focus on weak anomalies (small, low-contrast, or locally non-uniform textures). 

We also release our self-built dataset, AnoLCD, together with training and evaluation scripts to enable reproducible experiments. The table below reports the overall performance of **DLAS-Net** on **AnoLCD**.

| I-AUROC ↑ | I-AP ↑ | I-RPT@95TPR ↓ | P-AUROC ↑ | P-AP ↑ | P-RPT@95TPR ↓ |
|:---------:|:------:|:-------------:|:---------:|:------:|:-------------:|
| **99.4%** |**99.7%**| **2.6%**     | **98.6%** |  **88.0%**  | **4.4%** |


## **Highlight**

- Proposed a dual-level anomaly synthesis framework for LCD weak defect detection.
- Designed an image-level synthesis method to generate anomalies with diverse shapes and visibility
- Developed a feature-level synthesis method to produce subtle and weak anomalies
- Released the first LCD anomaly detection dataset for public benchmarking

## **Installation**

```bash
conda create -n dlas python=3.9 -y
conda activate dlas
pip install -r requirements.txt
 ```


## **Dataset**

### **Self-built datasets (AnoLCD)**


The dataset spans nine display screen categories, enabling comprehensive anomaly perception. It contains **2,450 normal** images and **750 anomalous** images. Of the normal images, **2,100** are used for training and **350** for testing.  
The anomaly set includes **250 spots**, **150 lines**, **50 mura**, **200 cylinders**, and **100 scratches**—**all anomaly images are used for testing**. Each image has a resolution of **1000×1000** pixels.

Representative examples are shown below: **(a)–(e)** correspond to **cylinder, line, mura, scratch,** and **spot** defects, respectively; **(f)** shows the **average proportion of pixels covered by defect masks** for each defect type. These defects primarily exhibit **low contrast, small size,** and **local non-uniformity**.

> **Provider & Download**  
> The dataset is provided by the Data Fusion Research Team at the University of Electronic Science and Technology of China.  
> 📥 **Download**: [Baidu Netdisk (extraction code: 2468)](https://pan.baidu.com/s/1y4ul30uijZ_oC5brcqEEfw?pwd=2468)


<img width="1005" height="729" alt="image" src="https://github.com/user-attachments/assets/44f8e0a6-6a71-4e1b-8cd1-2cea36db3d82" />






### **Other public datasets**


To evaluate the generalization of our method on both **weak anomalies** and **texture anomalies**, we conduct experiments on two public datasets:

- **MAD-sys** — A dataset targeting weak anomalies in complex industrial scenarios.  
  **Download:** https://www.mvtec.com/company/research/datasets/mvtec-ad/

- **MVTec-tex** — A texture benchmark (subset of MVTec-AD) for detecting and localizing defects in homogeneous textures.  
  **Download:** https://drive.google.com/file/d/1uLGWmOc4D9PuQawE-2nFS3p6XQzKrVsn/view?pli=1

> Please follow the licenses and usage terms of the respective datasets.



## **Run**


Open `shell/run.sh` and set your dataset paths and hyperparameters.

Launch training + evaluation:
```bash
bash run.sh
```

## **Acknowledge**
Thanks for the great inspiration from GLASS, SimpleNet!

## **Citation**




