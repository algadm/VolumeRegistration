import os
import torch
import argparse
import numpy as np
import nibabel as nib
import SimpleITK as sitk
from torchreg.metrics import NCC
from sklearn.model_selection import ParameterSampler
from utils import normalize_image, apply_transform_to_image, resample_to_match, get_corresponding_file
from torchreg import AffineRegistration
from metrics import NMI

def load_cbct_images(cbct_folder, seg_folder):
    """
    Retrieve the CBCT paths and their corresponding masks.

    Args:
        cbct_folder (str): Path to the folder containing the CBCTs.
        seg_folder (str): Path to the folder containing the segmentation masks.

    Returns:
        List[Tuple[str, str]]: List of tuples containing the CBCT path and mask path.
    """
    cbct_paths = []
    for filename in os.listdir(cbct_folder):
        if (filename.endswith(".nii.gz") or filename.endswith(".nii")):
            cbct_path = os.path.join(cbct_folder, filename)
            seg_path = get_corresponding_file(seg_folder, filename.split("_CBCT_")[0], "_Seg_")
            if os.path.exists(seg_path):
                cbct_paths.append((cbct_path, seg_path))
    return cbct_paths

def compute_mean_image(images):
    """
    Compute the voxel-wise mean of registered images.

    Args:
        images (List[Tuple[str, str]]): List of tuples containing the CBCT path and mask path.

    Returns:
        nib.Nifti1Image: Mean image
    """
    mean_data = np.mean([nib.load(img_path).get_fdata() for img_path, _ in images], axis=0)
    reference_image = nib.load(images[0][0])
    return nib.Nifti1Image(mean_data, reference_image.affine, reference_image.header)

def registration(mean, mean_path, cbct_path, seg_path, iter, patient_id, output_folder, device):
    """
    Register an individual CBCT to the current mean

    Args:
        mean (nib.Nifti1Image): Current mean image
        mean_path (str): Path to the mean image
        cbct_path (str): Path of the CBCT image
        seg_path (str): Path of the segmentation mask
        iter (int): Current iteration
        patient_id (str): Patient ID
        output_folder (str): Folder to save the registered image
        device (str): Device for computation (e.g., 'cuda' or 'cpu')

    Returns:
        str, nib.Nifti1Image: Registered image and it's path
    """
    image_folder = os.path.join(output_folder, f'iter_{iter}')
    os.makedirs(image_folder, exist_ok=True)
    registered_image_path = os.path.join(image_folder, f'{patient_id}_CBCT_registered.nii.gz')
    registered_mask_path = os.path.join(image_folder, f'{patient_id}_SEG_registered.nii.gz')
    
    param_grid = {
        'learning_rate': np.linspace(1e-4, 1e-3, 4),
        'sigma': np.linspace(5e-3, 5e-2, 3)
    }
    param_sampler = ParameterSampler(param_grid, n_iter=4)
    
    # resampled_moving = resample_to_match(mean_path, cbct_path)
    moving = torch.from_numpy(nib.as_closest_canonical(nib.load(cbct_path)).get_fdata()).float().to(device)
    static = torch.from_numpy(nib.as_closest_canonical(mean).get_fdata()).float().to(device)
    
    if iter == 0:
        print(seg_path)
        # resampled_mask_moving = resample_to_match(cbct_path, seg_path)
        mask_moving = torch.from_numpy(nib.as_closest_canonical(nib.load(seg_path)).get_fdata()).float().to(device)
        
        
        # TODO:
        # Get the corresponding segmentation mask for the mean image
        # resampled_mask_static = resample_to_match(mean_path, seg_path)
        mask_static = torch.from_numpy(nib.as_closest_canonical(nib.load("/home/lucia/Documents/Alban/data/CBCT_seg/B012_CBCT_Seg_Crop.nii.gz")).get_fdata()).float().to(device)
        
        moving *= mask_moving
        static *= mask_static
    
    moving_normed = (moving - moving.min()) / (moving.max() - moving.min() + 1e-8)
    static_normed = (static - static.min()) / (static.max() - static.min() + 1e-8)

    best_loss = float('inf')
    for params in param_sampler:
        print(f"\n\033[1mIteration: {iter} -- Using {device.upper()} -- Registering CBCT: {cbct_path} with mean: {mean_path}")
        print(f"Testing parameters: {params}\033[0m")
        
        nmi_loss_function = NMI(intensity_range=None, nbins=64, sigma=params['sigma'], use_mask=False)
        reg = AffineRegistration(scales=(4, 2), iterations=(500, 100), is_3d=True, 
                                 learning_rate=params['learning_rate'],
                                 dissimilarity_function=nmi_loss_function,
                                 optimizer=torch.optim.Adam,
                                 with_translation=True, with_rotation=True,
                                 with_zoom=False, with_shear=False,
                                 interp_mode="trilinear", padding_mode="zeros")
        moved_image = reg(moving_normed[None, None], static_normed[None, None])
        transform_matrix = reg.get_affine(with_grad=False)
        print(transform_matrix)
        
        loss = nmi_loss_function(moved_image, static_normed[None, None])
        print(f"Loss (NMI): {loss}")
        
        if loss < best_loss:
            best_loss = loss
            apply_transform_to_image(cbct_path, transform_matrix, registered_image_path, mean_path)
            # apply_transform_to_image(seg_path, transform_matrix, registered_mask_path, mean_path)
            
    print(f"Best loss: {best_loss}")

    return registered_image_path, registered_mask_path

def main(cbct_folder, seg_folder, output_folder, num_iterations):
    """
    Main function to compute the mean CBCT image iteratively.

    Args:
        cbct_folder (str): Path to the folder containing CBCT images.
        mask_folder (str): Path to the folder containing the segmentation masks.
        output_folder (str): Path to the folder where results will be saved.
        num_iterations (int): Number of iterations for computing the mean.
        param_sampler (ParameterSampler): Parameter sampler for registration.
    """
    if not os.path.isdir(cbct_folder): raise ValueError(f"CBCT folder does not exist: {cbct_folder}")
    
    os.makedirs(output_folder, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    cbct_paths = load_cbct_images(cbct_folder, seg_folder)
    mean_image = nib.load(cbct_paths[0][0])
    mean_path = os.path.join(output_folder, 'mean_image_initial.nii.gz')
    nib.save(mean_image, mean_path)

    for iteration in range(num_iterations):
        print(f"\nStarting iteration {iteration + 1}/{num_iterations}")
        registered_images = []

        for img_path, seg_path in cbct_paths:
            patient_id = os.path.basename(img_path).split("_CBCT_")[0]
            registered_image_path, registered_mask_path = registration(
                mean_image, mean_path, img_path, seg_path, iteration, patient_id, output_folder, device
            )
            registered_images.append((registered_image_path, registered_mask_path))

        mean_image = compute_mean_image(registered_images)
        mean_path = os.path.join(output_folder, f'mean_image_iter_{iteration + 1}.nii.gz')
        nib.save(mean_image, mean_path)
        print(f"Mean image for iteration {iteration + 1} saved to {mean_path}")

        cbct_paths = registered_images

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Creating CBCT mean for TMJ identification.')
    parser.add_argument('--cbct_folder', type=str, required=True, help='Path to the folder containing CBCT images')
    parser.add_argument('--seg_folder', type=str, required=True, help='Path to the folder containing the segmentation masks')
    parser.add_argument('--output_folder', type=str, required=True, help='Path to the folder where the mean will be saved')
    parser.add_argument('--num_iterations', type=int, default=5, help='Number of iterations for iterative mean computation')
    args = parser.parse_args()

    main(args.cbct_folder, args.seg_folder, args.output_folder, args.num_iterations)