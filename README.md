# Volume Registration

## Overview

`VolumeRegistration`  is a repository dedicated to exploring and implementing various methods for registering MRI to CBCT (Cone Beam Computed Tomography) images of the Temporomandibular joint (TMJ). Accurate volume registration is crucial in medical imaging for diagnostic and treatment planning purposes. This repository aims to provide multiple approaches, with at least one fully functional method, to perform MRI to CBCT registration.

## Different approaches

**1. Elastix using Normalized Mutual Information (NMI) Registration**
* **Status**: Functional but not yet fully optimized. Further investigation and refinement are needed. Although this method shows some results, I was unable to make it work properly for my current dataset using the provided parameters. The method does not perform adequately with my dataset, and I have moved on to more suitable approaches for my use case.
* **Description**: NMI is a robust metric commonly used in image registration to measure the statistical dependency between two images. It is particularly useful when registering multimodal images such as MRI and CBCT, where intensity values between the two images can differ significantly.
* **How to Use**: For now, the Elastix NMI registration method is available only via the terminal. To use it, navigate to the `VolumeRegistration` folder and execute the following command:
```
python3 ./NMI/nmi.py --cbct_folder path/to/cbct_folder --mri_folder path/to/mri_folder --output_folder path/to/output_folder
```

**2. Deep Learning-Based Registration (In Progress)**
* **Status**: Data separation into training, validation, and testing sets completed. A program was developped to be using [TorchIR](https://github.com/BDdeVos/TorchIR/tree/main) but there seems to be a problem with how the data is handled in 3D which makes it unusable for now.
* **Description**: This approach involves using a deep learning model to learn the transformation parameters directly from the data. The method aims to leverage the power of neural networks to achieve more accurate and efficient registration.


**3. TorchReg using NMI Registration**
* **Status**: Functional and close to fully optimized.
* **Description**: TorchReg uses Normalized Mutual Information (NMI) as the primary dissimilarity metric for MRI to CBCT registration. This technique is particularly effective for multimodal image registration, where intensity differences between images can vary significantly. The use of NMI ensures robust alignment by measuring the statistical dependence between the two images, making it ideal for medical imaging tasks. TorchReg seamlessly integrates with PyTorch workflows, providing flexibility and scalability for various image registration tasks. This method replaces the need for manual approximation steps previously required to complete a full registration.
* **Requirements**: To install the necessary libraries, navigate to the `TorchReg` folder and run the following command:
```
pip install -r requirements.txt
```

* **How to Use**: The TorchReg NMI registration method is currently available via the terminal. To use it, navigate to the `TorchReg` folder and run the following command:
```
python3 ./register.py --cbct_folder path/to/cbct_folder --mri_folder path/to/mri_folder --output_folder path/to/output_folder
```

**Note**: The NMI metric calculation was adapted from the work of [Bob D. de Vos](https://github.com/BDdeVos/TorchIR/blob/main/torchir/metrics.py), and credit goes to them for the implementation.