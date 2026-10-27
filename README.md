# CIFAR-10-image-classification
CIFAR-10 image classification: a CNN  vs. ResNet50 transfer learning with fine-tuning.

Image classification on the CIFAR-10 dataset, comparing a CNN built from scratch with a ResNet50 model fine-tuned with transfer learning.

## Overview
CIFAR-10 images are only 32x32 pixels, which makes similar classes (cat/dog, deer/horse, automobile/truck) hard to separate. This project builds a CNN and then applies ResNet50 transfer learning to see how much a pretrained network improves the result. Images are upscaled to 128x128 inside the model so the pretrained filters work at a scale they know. The model was trained in two stages: first the new classification head, then the last ResNet block fine-tuned with a low learning rate.

## Approach
1. Loaded the data and checked the class balance
2. Built a balanced training subset with a fixed random seed
3. ** CNN:** 3 conv blocks (Conv2D, BatchNormalization, MaxPooling), dropout, dense head, with flip and shift augmentation
4. **ResNet50:** ImageNet weights, `Resizing(128, 128)`, global average pooling, dropout and a dense head
   - Stage 1: base frozen, head trained with Adam at 1e-3
   - Stage 2: last block (`conv5`) unfrozen and fine-tuned with Adam at 1e-5
5. Used early stopping with best weights restored
6. Evaluated both models on the official test set

## Results (test set, 10,000 images)
| Model | Test accuracy |
|-------|---------------|
| CNN (from scratch) | 82% |
| ResNet50 (transfer learning) | 93.09% |

### ResNet50 per-class results
| Class | Precision | Recall | F1-score |
|-------|-----------|--------|----------|
| airplane | 0.952 | 0.943 | 0.947 |
| automobile | 0.979 | 0.943 | 0.961 |
| bird | 0.945 | 0.916 | 0.930 |
| cat | 0.857 | 0.872 | 0.864 |
| deer | 0.932 | 0.903 | 0.917 |
| dog | 0.894 | 0.871 | 0.882 |
| frog | 0.944 | 0.959 | 0.951 |
| horse | 0.927 | 0.958 | 0.942 |
| ship | 0.953 | 0.970 | 0.961 |
| truck | 0.929 | 0.974 | 0.951 |
| **Macro avg** | **0.931** | **0.931** | **0.931** |

## Tech Stack
Python, TensorFlow, Keras, Scikit-learn, NumPy, Matplotlib



