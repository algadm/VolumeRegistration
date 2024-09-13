# Volume Registration

## Overview

`VolumeRegistration`  is a repository dedicated to exploring and implementing various methods for registering MRI to CBCT (Cone Beam Computed Tomography) images. Accurate volume registration is crucial in medical imaging for diagnostic and treatment planning purposes. This repository aims to provide multiple approaches, with at least one fully functional method, to perform MRI to CBCT registration.

## Different approaches

**1. Elastix using Normalized Mutual Information (NMI) Registration**
* **Status**: Functional but not yet fully optimized. Further investigation and refinement are needed.
* **Description**: NMI is a robust metric commonly used in image registration to measure the statistical dependency between two images. It is particularly useful when registering multimodal images such as MRI and CBCT, where intensity values between the two images can differ significantly.
* **How to Use**: For now, the Elastix NMI registration method is available only via the terminal. To use it, navigate to the `VolumeRegistration` folder and execute the following command:
```
python3 ./NMI/nmi.py --cbct_folder path/to/cbct_folder --mri_folder path/to/mri_folder --output_folder path/to/output_folder
```

**2. Deep Learning-Based Registration (In Progress)**
* **Status**: Data separation into training, validation, and testing sets completed. A program was developped to be using [TorchIR](https://github.com/BDdeVos/TorchIR/tree/main) but there seems to be a problem with how the data is handled in 3D which makes it unusable for now.
* **Description**: This approach involves using a deep learning model to learn the transformation parameters directly from the data. The method aims to leverage the power of neural networks to achieve more accurate and efficient registration.


**3. TorchReg using NMI Registration**
* **Status**: Functional and close to fully optimized. While it performs well in most cases, further refinement is needed for edge cases.
* **Description**: TorchReg leverages Normalized Mutual Information (NMI) as the dissimilarity metric for MRI to CBCT registration. This approach is particularly effective for multimodal image registration, where intensity values between the images differ significantly. The use of NMI ensures robust alignment by measuring the statistical dependence between the images, making it well-suited for medical imaging tasks. TorchReg provides a flexible and scalable solution, allowing for seamless integration with PyTorch-based workflows.
* **How to Use**: The TorchReg NMI registration method is currently available via the terminal. To use it, first install the required package:
```
pip install torchreg
```

Once the package is installed, navigate to the VolumeRegistration folder and run the following command:
```
python3 ./TorchReg/register.py --cbct_folder path/to/cbct_folder --mri_folder path/to/mri_folder --output_folder path/to/output_folder
```