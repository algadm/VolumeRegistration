import os
import numpy as np
import nibabel as nib
import SimpleITK as sitk

def normalize_image(image):
    """
    Normalize the image intensities to the range [0, 1]

    Args:
        image (nib.Nifti1Image): Image to normalize

    Returns:
        nib.Nifti1Image: Normalized image
    """
    image_data = image.get_fdata()
    norm_data = (image_data - image_data.min()) / (image_data.max() - image_data.min() + 1e-8)
    return nib.Nifti1Image(norm_data, image.affine, image.header)

def get_corresponding_file(folder, patient_id, modality):
    """
    Retrieve the corresponding file for a given patient ID and modality.

    Args:
        folder (str): Path to the folder containing the files.
        patient_id (str): Unique identifier for the patient.
        modality (str): Modality keyword to match in the file name (e.g., '_MR_').

    Returns:
        str or None: Full path to the corresponding file if found, otherwise None.
    """
    for root, _, files in os.walk(folder):
        for file in files:
            if file.startswith(patient_id) and modality in file and (file.endswith(".nii.gz") or file.endswith(".nii")):
                return os.path.join(root, file)
    return None

def apply_transform_to_image(image_path, transform_matrix, output_path, default_pixel_value=0):
    """
    Apply an affine transformation to an image and save the transformed image.

    Args:
        image_path (str): Path to the input image.
        transform_matrix (torch.Tensor): Transformation matrix from registration.
        output_path (str): Path to save the transformed image.
    """
    try:
        image = sitk.ReadImage(image_path)
    except Exception as e:
        print(f"Error reading image {image_path}: {e}")
        return
    
    image_np = sitk.GetArrayFromImage(image)
    normalized_np = (image_np - image_np.min()) / (image_np.max() - image_np.min() + 1e-8)
    normalized_image = sitk.GetImageFromArray(normalized_np)
    normalized_image.CopyInformation(image)

    transform = sitk.AffineTransform(3)
    affine_np = transform_matrix.squeeze(0).cpu().numpy()

    if affine_np.shape == (3, 4):
        affine_np = np.vstack([affine_np, [0, 0, 0, 1]])

    transform.SetMatrix(affine_np[:3, :3].flatten())
    transform.SetTranslation(affine_np[:3, 3])

    resample = sitk.ResampleImageFilter()
    resample.SetReferenceImage(normalized_image)
    resample.SetTransform(transform)
    resample.SetInterpolator(sitk.sitkLinear)
    resample.SetDefaultPixelValue(default_pixel_value)
    resample.SetOutputPixelType(sitk.sitkFloat32)

    transformed_image = resample.Execute(normalized_image)
    sitk.WriteImage(transformed_image, output_path)
    print(f"Transformed and normalized image saved to {output_path}")