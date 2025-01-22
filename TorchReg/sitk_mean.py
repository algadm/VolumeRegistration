import os
import torch
import argparse
import numpy as np
import nibabel as nib
import SimpleITK as sitk
from torchreg.metrics import NCC
from sklearn.model_selection import ParameterSampler
from utils import normalize_image, apply_transform_to_image
from torchreg import AffineRegistration
from metrics import NMI

def load_cbct_images(cbct_folder):
    """
    Load all images in the folder and normalize them

    Args:
        cbct_folder (str): Path to the folder containing the CBCTs

    Returns:
        List[Tuple[str, nib.Nifti1Image]]: List of tuples containing both the normalized NIfTI image and its path
    """
    cbct_images = []
    for filename in os.listdir(cbct_folder):
        if (filename.endswith(".nii.gz") or filename.endswith(".nii")):
            file_path = os.path.join(cbct_folder, filename)
            img = nib.load(file_path)
            norm_img = normalize_image(img)
            cbct_images.append((file_path, norm_img))
    return cbct_images

def compute_mean_image(images):
    """
    Compute the voxel-wise mean of images

    Args:
        images (List[Tuple[str, nib.Nifti1Image]]): List of input images and their paths

    Returns:
        nib.Nifti1Image: Mean image
    """
    mean_data = np.mean([img.get_fdata() for _, img in images], axis=0)
    reference_image = images[0][1]
    return nib.Nifti1Image(mean_data, reference_image.affine, reference_image.header)

def registration(mean, cbct_image, cbct_path, iter, patient_id, output_folder):
    """
    Register an individual CBCT to the current mean using SimpleITK.

    Args:
        mean (nib.Nifti1Image): Current mean image
        cbct_image (nib.Nifti1Image): CBCT image to register
        cbct_path (str): Path of the CBCT image
        iter (int): Current iteration
        patient_id (str): Patient ID
        output_folder (str): Folder to save the registered image

    Returns:
        str, nib.Nifti1Image: Registered image and its path
    """
    mean_sitk = sitk.GetImageFromArray(mean.get_fdata())
    mean_sitk.SetSpacing(list(mean.header.get_zooms()))
    mean_sitk.SetDirection(mean.affine[:3, :3].flatten())
    mean_sitk.SetOrigin(mean.affine[:3, 3])

    cbct_sitk = sitk.GetImageFromArray(cbct_image.get_fdata())
    cbct_sitk.SetSpacing(list(cbct_image.header.get_zooms()))
    cbct_sitk.SetDirection(cbct_image.affine[:3, :3].flatten())
    cbct_sitk.SetOrigin(cbct_image.affine[:3, 3])

    registration_method = sitk.ImageRegistrationMethod()
    transform = sitk.CenteredTransformInitializer(
        mean_sitk, cbct_sitk, sitk.Euler3DTransform(), sitk.CenteredTransformInitializerFilter.MOMENTS
    )
    registration_method.SetInitialTransform(transform, inPlace=False)

    registration_method.SetMetricAsMeanSquares()
    registration_method.SetOptimizerAsGradientDescent(learningRate=1.0, numberOfIterations=100)

    final_transform = registration_method.Execute(mean_sitk, cbct_sitk)
    resampled_cbct = sitk.Resample(
        cbct_sitk, mean_sitk, final_transform, sitk.sitkLinear, 0.0, cbct_sitk.GetPixelID()
    )
    registered_image_folder = os.path.join(output_folder, f'iter_{iter}')
    os.makedirs(registered_image_folder, exist_ok=True)
    registered_image_path = os.path.join(registered_image_folder, f'{patient_id}_CBCT_registered.nii.gz')
    sitk.WriteImage(resampled_cbct, registered_image_path)

    registered_image = nib.Nifti1Image(
        sitk.GetArrayFromImage(resampled_cbct),
        affine=np.eye(4)
    )

    print(f"Registered image saved to {registered_image_path}")
    return registered_image_path, registered_image

def main(cbct_folder, output_folder, num_iterations, param_sampler):
    """
    Main function to compute the mean CBCT image iteratively.

    Args:
        cbct_folder (str): Path to the folder containing CBCT images.
        output_folder (str): Path to the folder where results will be saved.
        num_iterations (int): Number of iterations for computing the mean.
        param_sampler (ParameterSampler): Parameter sampler for registration.
    """
    os.makedirs(output_folder, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    cbct_images = load_cbct_images(cbct_folder)
    mean_image = cbct_images[0][1]
    mean_path = os.path.join(output_folder, 'mean_image_initial.nii.gz')
    nib.save(mean_image, mean_path)

    for iteration in range(num_iterations):
        print(f"\nStarting iteration {iteration + 1}/{num_iterations}")
        registered_images = []

        for path, img in cbct_images:
            patient_id = os.path.basename(path).split("_CBCT_")[0]
            registered_image_path, registered_image = registration(
                mean_image, img, path, iteration, patient_id, output_folder
            )
            registered_images.append((registered_image_path, registered_image))

        mean_image = compute_mean_image(registered_images)
        mean_path = os.path.join(output_folder, f'mean_image_iter_{iteration + 1}.nii.gz')
        nib.save(mean_image, mean_path)
        print(f"Mean image for iteration {iteration + 1} saved to {mean_path}")

        cbct_images = registered_images

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Creating CBCT mean for TMJ identification.')
    parser.add_argument('--cbct_folder', type=str, required=True, help='Path to the folder containing CBCT images')
    parser.add_argument('--output_folder', type=str, required=True, help='Path to the folder where the mean will be saved')
    parser.add_argument('--num_iterations', type=int, default=5, help='Number of iterations for iterative mean computation')
    
    args = parser.parse_args()

    param_grid = {'learning_rate_rigid': np.logspace(-3, -2, 4)}
    param_sampler = ParameterSampler(param_grid, n_iter=4)

    main(args.cbct_folder, args.output_folder, args.num_iterations, param_sampler)