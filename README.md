# **DLAS-Net:Dual-Level Anomaly Synthesis Network for Weak Anomaly Detection in Liquid Crystal Displays**

![Fig2_框架图3](https://github.com/user-attachments/assets/d7986cac-8d1e-4297-885b-9faca3d803d8)



## **Introduction**
This repo contains the official PyTorch implementation of DLAS-Net, a dual-level anomaly synthesis network for self-supervised anomaly detection in display, with a particular focus on weak anomalies (small, low-contrast, or locally non-uniform textures). 
We also release our self-built dataset, AnoLCD, together with training and evaluation scripts to enable reproducible experiments.

### Results on AnoLCD

The table below reports the overall performance of **DLAS-Net** on **AnoLCD**.

| I-AUROC ↑ | I-AP ↑ | I-RPT@95TPR ↓ | P-AUROC ↑ | P-AP ↑ | P-RPT@95TPR ↓ |
|:---------:|:------:|:-------------:|:---------:|:------:|:-------------:|
| **99.4%** |**99.7%**| **2.6%**     | **98.6%** |  **88.0%**  | **4.4%** |


## **Highlight**

- Proposed a dual-level anomaly synthesis framework for LCD weak defect detection.
- Designed an image-level synthesis method to generate anomalies with diverse shapes and visibility
- Developed a feature-level synthesis method to produce subtle and weak anomalies
- Released the first LCD anomaly detection dataset for public benchmarking

## **Installation**

Our codebase is based on [YOLOv5]([https://github.com/facebookresearch/detectron2](https://github.com/ultralytics/yolov5)). You only need to follow its instructions for installation.

## **Dataset Preparation**

### **LCD light defect dataset**

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

## **Result**
### **Result on LCDLD dataset**

| Models    | P (M) | P (L) | P (S) | R (M) | R (L) | R (S) | AP (M) | AP (L) | AP (S) | mAP  | Params | FLOPs |
|-----------|-------|-------|-------|-------|-------|-------|--------|--------|--------|------|--------|-------|
| YOLOv5s   | 99.8  | 89.9  | 94.7  | 100   | 83.3  | 95.0  | 99.5   | 85.6   | 94.9   | 93.3 | 7.0    | 16.0  |
| YOLOXs    | 99.4  | 86.8  | 89.7  | 100   | 82.2  | 86.3  | 99.5   | 94.4   | 90.2   | 91.4 | 10.6   | 23.6  |
| YOLOv6s   | 99.5  | 66.2  | 94.4  | 100   | 87.8  | 79.3  | 99.5   | 86.4   | 91.9   | 92.6 | 16.3   | 44.2  |
| YOLOv7    | 98.9  | 91.9  | 92.7  | 100   | 88.9  | 93.3  | 99.5   | 88.2   | 93.8   | 93.8 | 37.2   | 105.2 |
| YOLOv8s   | 99.2  | 78.9  | 95.1  | 100   | 77.8  | 87.1  | 99.5   | 84.0   | 95.0   | 92.8 | 11.1   | 28.6  |
| YOLOv9s   | 99.9  | 83.4  | 95.8  | 100   | 82.2  | 86.8  | 99.5   | 85.3   | 94.9   | 93.3 | 9.6    | 38.7  |
| YOLOv10s  | 98.8  | 90.6  | 87.9  | 97.2  | 84.4  | 90.8  | 99.4   | 91.1   | 92.5   | 94.3 | 8.0    | 24.5  |
| YOLOv11s  | 99.8  | 91.5  | 94.3  | 100   | 83.3  | 90.0  | 99.5   | 89.5   | 94.2   | 94.4 | 9.4    | 21.3  |
| Ours      | 100.0 | 93.0  | 92.8  | 100   | 93.2  | 96.2  | 99.5   | 94.8   | 95.7   | 96.7 | 7.4    | 20.3  |

The model weight files can be downloaded at: https://pan.baidu.com/s/1ECJpvRn4xe-UCIrBAuGTCg?pwd=1357.
### **Result on PKU-Market-Phone dataset**
| Models       | P (O) | P (SC) | P (ST) | R (O) | R (SC) | R (ST) | AP (O) | AP (SC) | AP (ST) | mAP  |
|--------------|-------|--------|--------|-------|--------|--------|--------|---------|---------|------|
| YOLOv5s      | 98.3  | 96.4   | 97.0   | 98.8  | 95.6   | 97.2   | 98.6   | 96.7    | 96.2    | 97.2 |
| YOLOXs       | 98.4  | 96.8   | 97.1   | 97.0  | 87.3   | 93.4   | 98.9   | 96.2    | 96.4    | 96.2 |
| YOLOv6s      | 97.1  | 94.0   | 95.3   | 97.6  | 95.8   | 88.1   | 98.9   | 97.0    | 94.3    | 96.7 |
| YOLOv7       | 97.7  | 96.5   | 98.3   | 98.8  | 96.3   | 97.6   | 98.8   | 96.3    | 97.6    | 97.6 |
| YOLOv8s      | 97.6  | 91.3   | 94.7   | 99.2  | 97.1   | 89.1   | 99.1   | 97.8    | 94.6    | 97.2 |
| YOLOv9s      | 97.2  | 96.3   | 94.6   | 98.2  | 95.0   | 86.5   | 98.9   | 97.1    | 95.5    | 97.2 |
| YOLOv10s     | 93.9  | 93.7   | 95.1   | 96.4  | 94.4   | 85.6   | 98.0   | 96.9    | 94.6    | 96.5 |
| YOLOv11s     | 95.8  | 92.7   | 92.7   | 98.8  | 96.8   | 91.8   | 99.2   | 97.6    | 94.9    | 97.2 |
| Ours         | 99.4  | 95.0   | 97.8   | 98.8  | 95.8   | 98.3   | 99.3   | 97.6    | 98.4    | 98.5 |

The model weight files can be downloaded at: https://pan.baidu.com/s/1dDz_-8PBU_B9IYvf89bs_g?pwd=1357.
### **Result on PKU-Market-PCB datasett**

| Metrics  | YOLOv5s | YOLOXs | YOLOv6s | YOLOv7 | YOLOv8s | YOLOv9s | YOLOv10s | YOLOv11s | Ours |
|--------- |---------|--------|---------|--------|---------|---------|----------|----------|------|
| P (Mh)   | 98.8    | 98.4   | 98.4    | 91.7   | 99.1    | 99.1    | 96.5     | 97.7     | 98.5 |
| P (Mb)   | 91.5    | 95.9   | 82.5    | 82.1   | 93.7    | 94.0    | 96.7     | 93.8     | 92.8 |
| P (Oc)   | 95.4    | 95.9   | 92.0    | 93.5   | 95.0    | 96.3    | 94.8     | 95.4     | 97.3 |
| P (Sh)   | 97.4    | 98.2   | 95.4    | 96.5   | 94.8    | 95.1    | 95.8     | 95.7     | 96.1 |
| P (Sp)   | 96.3    | 96.3   | 85.6    | 94.1   | 98.2    | 95.1    | 97.7     | 95.1     | 95.8 |
| P (Sc)   | 91.2    | 93.8   | 83.9    | 96.2   | 97.5    | 97.3    | 93.1     | 89.0     | 98.0 |
| R (Mh)   | 99.1    | 99.1   | 98.2    | 98.9   | 99.5    | 98.6    | 97.3     | 99.1     | 99.1 |
| R (Mb)   | 90.4    | 78.3   | 78.3    | 83.1   | 86.7    | 91.3    | 94.4     | 86.7     | 96.4 |
| R (Oc)   | 98.1    | 80.0   | 81.0    | 84.6   | 89.5    | 92.0    | 92.9     | 97.7     | 100  |
| R (Sh)   | 96.6    | 96.0   | 86.2    | 93.8   | 95.0    | 97.4    | 91.4     | 95.5     | 97.4 |
| R (Sp)   | 81.4    | 76.0   | 82.7    | 73.5   | 71.6    | 83.3    | 84.3     | 76.3     | 85.3 |
| R (Sc)   | 96.0    | 79.2   | 80.2    | 82.2   | 89.1    | 87.9    | 92.1     | 89.1     | 97.2 |
| AP (Mh)  | 99.3    | 99.1   | 98.9    | 98.7   | 99.4    | 98.8    | 99.0     | 99.0     | 98.8 |
| AP (Mb)  | 89.8    | 90.8   | 83.1    | 87.6   | 92.4    | 95.3    | 93.4     | 92.7     | 96.5 |
| AP (Oc)  | 98.5    | 85.9   | 90.8    | 90.9   | 95.2    | 94.8    | 95.7     | 99.1     | 99.4 |
| AP (Sh)  | 99.2    | 97.8   | 94.5    | 95.2   | 97.6    | 98.5    | 96.3     | 98.0     | 97.8 |
| AP (Sp)  | 86.4    | 80.9   | 77.7    | 78.6   | 88.7    | 87.3    | 85.6     | 86.7     | 85.3 |
| AP (Sc)  | 97.1    | 88.8   | 85.2    | 92.1   | 93.1    | 98.0    | 94.1     | 92.4     | 98.2 |
| mAP      | 95.2    | 90.6   | 88.3    | 90.5   | 94.4    | 95.4    | 94.0     | 94.7     | 97.1 |


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

