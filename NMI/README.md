# Volume Registration

## Overview

`VolumeRegistration`  is a repository dedicated to exploring and implementing various methods for registering MRI to CBCT (Cone Beam Computed Tomography) images. Accurate volume registration is crucial in medical imaging for diagnostic and treatment planning purposes. This repository aims to provide multiple approaches, with at least one fully functional method, to perform MRI to CBCT registration.

## Different approaches

**1. Normalized Mutual Information (NMI) Registration**
* **Status**: Functional but not yet fully optimized. Further investigation and refinement are needed.
* **Description**: NMI is a robust metric commonly used in image registration to measure the statistical dependency between two images. It is particularly useful when registering multimodal images such as MRI and CBCT, where intensity values between the two images can differ significantly.
* **How to Use**: For now, the NMI registration method is available only via the terminal. To use it, navigate to the `VolumeRegistration` folder and execute the following command:
```
python3 ./NMI/nmi.py --cbct_folder path/to/cbct_folder --mri_folder path/to/mri_folder --output_folder path/to/output_folder
```

**2. Deep Learning-Based Registration (In Progress)**
* **Status**: Data separation into training, validation, and testing sets completed.
* **Description**: This approach involves using a deep learning model to learn the transformation parameters directly from the data. The method aims to leverage the power of neural networks to achieve more accurate and efficient registration.