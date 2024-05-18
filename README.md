# EEGMobile

Accepted HCII 2024: PENDING

Abstract

Electroencephalography (EEG) analysis is an important domain in the realm of Brain-Computer interface (BCI) research. To ensure BCI devices are capable of providing practical applications in the real world, brain signal processing techniques must be fast, accurate, and resource-conscious to deliver low-latency neural analytics. As a result, this study proposes the use of a lightweight, Transformer-based network for EEG gaze estimation. This approach involves leveraging a pre-trained MobileViT model alongside a Knowledge Distillation-based (KD) training procedure to develop a model with a balance of speed, accuracy, and size for EEG Eye Tracking (ET) tasks. Our results showcase this model is capable of performing at a level comparable (only 3\% lower) to the previous State-Of-The-Art (SOTA) on the EEGEyeNet Absolute Position Task while being 33\% faster and 60\% smaller than the SOTA. Our research presents a cost-effective model with applications, particularly on resource-constrained devices, and sets a baseline for expanding future research on lightweight, mobile-friendly models for EEG regression.

# Overview
EEGMobile incorporates a pretrained MobileViT network first presented by Mehta & Rastegari in: ["MobileViT: Light-weight, General-purpose, and Mobile-friendly Vision Transformer
"](https://arxiv.org/abs/2110.02178) and further expanded in ["Separable Self-attention for Mobile Vision Transformers"](https://arxiv.org/abs/2206.02680). Furthermore, this model utilized Knowledge Distillation in the training procedure, based on the work of Hinton et al. in: ["Distilling the Knowledge in a Neural Network"](https://arxiv.org/abs/1503.02531). 

This repository inlcudes the original EEGViT models which can be found [here](https://github.com/ruiqiRichard/EEGViT), the EEGViT-TCNet (teacher model) which can be found [here](https://github.com/ModeEric/EEGViT-TCNet), and our EEGMobile model. Weights for pretrained models were loaded from [huggingface.co](https://huggingface.co/).

# Dataset
Data for the EEGEyeNet Absolute Position Task can be downloaded with
```
wget -O "./dataset/Position_task_with_dots_synchronised_min.npz" "https://osf.io/download/ge87t/"
```
More information on this dataset and others can be found in: ["EEGEyeNet: a Simultaneous Electroencephalography and Eye-tracking Dataset and Benchmark for Eye Movement Prediction"](https://arxiv.org/abs/2111.05100)

# Requirements
Basic requirements can be installed with
```
pip install -r general_requirements.txt
```

# Basic Usage
Default training of the teacher model (EEGViT-TCNet) or others can be done with
```
python run.py
```

Once the teacher model's weights have been saved, they can be loaded to train EEGMobile with
```
python distillation_run.py
```

You can load and run a speed test on any saved model using
```
python inference_test.py
```

Be sure you have selected the right model when loading saved weights.
