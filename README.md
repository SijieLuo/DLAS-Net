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

conda create -n dlas python=3.9 -y
conda activate dlas
pip install -r requirements.txt


## **Dataset Preparation**

### **LCD dataset**

The LCD light defect dataset consists of images displayed on a 7-inch screen with a resolution of 768×1280 pixels, covering spot, line, and mura defects. It includes 1,608 images in total. Due to the limited number of actual defects available during production, data augmentation was applied, resulting in 225 spot, 483 line, and 900 mura. The dataset is provided by the data fusion research team at the University of Electronic Science and Technology of China. To download the dataset, please visit: https://pan.baidu.com/s/1R7OENxkxPrY5RVweAtToPg?pwd=1357

Samples

![lcd_simples](https://github.com/user-attachments/assets/e5946377-02dd-4e98-8321-83884c2d0b23)

### **LCD surface defect dataset**

The surface defect dataset included three types of defects: oil, scratches and stains, with 400 images per defect type at a resolution of 1920×1080. The dataset is built and presented by Jian Zhang, Miaoju Ban (Open Lab on Human Robot Interaction, Peking University). To download the dataset, please visit: https://robotics.pkusz.edu.cn/resources/dataset/.

Samples

<p float="left">
  <img src="https://github.com/SijieLuo/Detail-aware-network-with-contrast-attention/assets/52660906/91ef6aa7-a645-4562-8274-2ae2c0174657" width="32%" />
  <img src="https://github.com/SijieLuo/Detail-aware-network-with-contrast-attention/assets/52660906/f371add6-8acc-4867-ab6a-af10aaf2bffa" width="32%" />
  <img src="https://github.com/SijieLuo/Detail-aware-network-with-contrast-attention/assets/52660906/0cfcba51-7b01-4e1a-819e-11bed5b57b81" width="32%" />
</p>

<p align="center">
  <span>oil</span>&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;<span>scratch</span>&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;<span>stain</span>
</p>

### **PCB surface defect dataset**

The PCB defect dataset contained 693 images with six types of defects: missing holes, open circuit, mouse bites, spur, short, and spurious copper. The dataset is built and presented by Lihui Dai et al. (Open Lab on Human Robot Interaction, Peking University). To download the dataset, please visit: https://robotics.pkusz.edu.cn/resources/dataset/.

Samples

![绘图2](https://github.com/SijieLuo/Detail-aware-network-with-contrast-attention/assets/52660906/bfa1176f-084a-4302-aa58-ea99bde3b24d)




The model weight files can be downloaded at: https://pan.baidu.com/s/1zj2D1yZ1SHY-j2yJOWyEZg?pwd=1357.

## **Acknowledge**
The code base is built with ultralytics. Thanks for the great implementations!

## **Citation**
Please cite the following paper if the code and dataset help your project:

```bibtex
@article{luo2024daca,
  title={DACA-Net: Detail-aware network with contrast attention for locating liquid crystal display defects},
  author={Luo, Sijie and Chen, Huaixin and Liu, Biyuan},
  journal={Displays},
  pages={102913},
  year={2024},
  publisher={Elsevier}
}
```

