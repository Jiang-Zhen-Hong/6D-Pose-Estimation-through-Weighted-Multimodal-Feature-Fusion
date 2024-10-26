# 6D-Pose-Estimation-through-Weighted-Multimodal-Feature-Fusion

This is a 6D object pose estimation network via weighted feature fusion of RGB, point cloud, mask, and normal vector.

# Compiling

The program was trained and tested under a 64-bit Linux distribution.(Ubuntu 20.04)  

# Training

LineMOD Dataset:  
Download and unzip the Linemod_preprocessed.zip, then run: ./experiments/scripts/train_linemod.sh  
YCB_Video Dataset:  
Download and unzip the YCB_Video_Dataset.zip, then run: ./experiments/scripts/train_ycb.sh  

# Evaluation

LineMOD Dataset:  
Run: ./experiments/scripts/eval_linemod.sh  
YCB_Video Dataset:  
Run: ./experiments/scripts/eval_ycb.sh  

# Usage

The code is mainly based on DenseFusion(https://github.com/j96w/DenseFusion), MaskedFusion(https://github.com/kroglice/MaskedFusion) and normalSpeed(https://github.com/hfutcgncas/normalSpeed)  
The dataset can be download at:  
[LineMOD Dataset](https://hkustconnect-my.sharepoint.com/:u:/g/personal/yhebk_connect_ust_hk/ETW6iYHDbo1OsIbNJbyNBkABF7uJsuerB6c0pAiiIv6AHw?e=eXM1UE)  
[YCB_Video Dataset](https://rse-lab.cs.washington.edu/projects/posecnn/)  
