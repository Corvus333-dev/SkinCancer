# Teaching AI to Spot Skin Cancer

![Cartoon header](assets/lesion_cartoon.png)

> Transfer learning ensemble that leverages dermoscopic image features and patient metadata via multimodal fusion for 
> multi-class skin lesion diagnosis.

## Introduction

In about four minutes, someone will die from skin cancer. It's a disease that hides in plain sight—visible to the naked
eye, yet often overlooked until it's too late. The cruel irony is that most of these cases are preventable, as the clues 
are literally right on the surface.

This project builds a deep learning tool for screening skin lesions. Two lightweight models 
(EfficientNetB1 and ResNet50) learn from dermoscopy images paired with patient metadata, each generating independent 
predictions. An ensemble merges these into a consensus diagnosis: faster than a human, more consistent than either model 
alone, and aware of its uncertainty.