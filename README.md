# Teaching AI to Spot Skin Cancer

![Lesion cartoon](assets/lesion_cartoon.png)

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

## Dataset

The HAM10000 dataset consists of 10015 dermoscopic images of biologically diverse skin lesions:

| Lesion Type                                   | Label | Category          |
|-----------------------------------------------|-------|-------------------|
| Actinic Keratosis / Intraepithelial Carcinoma | akiec | ⚠️ Premalignant   |
| Basal Cell Carcinoma                          | bcc   | ❌ Malignant       |
| Benign Keratoses                              | bkl   | ✅ Benign          |
| Dermatofibroma                                | df    | ✅ Benign          |
| Melanocytic Nevi                              | nv    | ✅ Benign          |
| Melanoma                                      | mel   | ❌ Malignant       |
| Vascular Lesions                              | vasc  | ⚠️ Usually Benign |

### Partial Balancing Effect:

![Dataset distribution](data/plots/dx_dist.png)
> Due to extreme class imbalance, melanocytic nevi were undersampled by removing their duplicate images, whereas 
> duplicate images from minority classes were retained in the training set. This preserves lesion-level integrity across 
> splits while leveraging minority duplicates as natural augmentation.

### Lesion Visualization:

![Lesion images](data/plots/lesions.png)
> Sample images arranged by the ensemble model's predicted probability for the ground-truth 
> diagnosis. Rows correspond to high, medium, and low confidence cases, revealing morphological 
> heterogeneity and task difficulty (even for dermatologists!).

### Citation

Tschandl, P. (2018). The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented 
skin lesions (Version V4) [dataset]. Harvard Dataverse. https://doi.org/10.7910/DVN/DBW86T