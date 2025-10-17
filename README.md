# Teaching AI to Spot Skin Cancer

![Cartoon header](assets/lesion_cartoon.png)

> A multimodal convolutional neural network (CNN) ensemble that learns from dermoscopy images and patient metadata to detect
> and classify skin lesions.

## Introduction

In about four minutes, someone will die from skin cancer. It's a disease that hides in plain sight; visible to the naked
eye, yet often overlooked until it's too late. The tragedy is that most of these cases are preventable, as the clues 
are—literally—right on the surface.

This project explores how machine learning can lend a sharper lens. By combining dermoscopy images with conventional 
patient metadata, I train CNNs to answer a question every dermatologist faces: _What am I looking at?_ Two lightweight 
models (EfficientNetB1 and ResNet50) learn in tandem, each producing independent predictions. An ensemble averages these 
outputs, providing a digital second option that improves reliability and maintains real-world screening efficiency.