import os
import torch
import argparse
import numpy as np
import nibabel as nib
from metrics import NMI
from torchreg import AffineRegistration
from sklearn.model_selection import ParameterSampler
from utils import get_corresponding_file

def save_as_nifti(moving_path, static_path, output_path):
    """
    Save the registered image as a NIfTI file.

    Args:
        moving_path (str): Path to the moving image.
        static_path (str): Path to the static image.
        output_path (str): Path to save the registered image.
    """
    static_nifti = nib.load(static_path)
    new_nifti = nib.Nifti1Image(moving_path.cpu().numpy(), static_nifti.affine, static_nifti.header)
    nib.save(new_nifti, output_path)
    print("Saved registered image to:", output_path)

def main(cbct_folder, mri_folder, output_folder):
    """
    Main function to perform registration of CBCT images with corresponding MRI images.

    Args:
        cbct_folder (str): Path to the folder containing CBCT images.
        mri_folder (str): Path to the folder containing MRI images.
        output_folder (str): Path to save the registered images.
        param_sampler (ParameterSampler): Sampler for generating hyperparameter combinations.
    """
    os.makedirs(output_folder, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    param_grid = {
        'learning_rate': np.logspace(-5, -3, 10),
        'sigma': np.logspace(-3, -2, 3)
    }
    param_sampler = ParameterSampler(param_grid, n_iter=30)

    if not os.path.isdir(cbct_folder): raise ValueError(f"CBCT folder does not exist: {cbct_folder}")
    if not os.path.isdir(mri_folder): raise ValueError(f"MRI folder does not exist: {mri_folder}")

    for root, _, files in os.walk(cbct_folder):
        for cbct_file in files:
            if "_CBCT_" in cbct_file and (cbct_file.endswith(".nii") or cbct_file.endswith(".nii.gz")):
                patient_id = cbct_file.split("_CBCT_")[0]
                cbct_path = os.path.join(root, cbct_file)
                mri_path = get_corresponding_file(mri_folder, patient_id, "_MR_")

                if not mri_path:
                    print(f"No corresponding MRI file found for: {cbct_file}")
                    continue
                
                output_path = os.path.join(output_folder, f'{patient_id}_MR_registered.nii.gz')

                moving_nii = nib.load(mri_path)
                static_nii = nib.load(cbct_path)

                # Convert to PyTorch tensors
                moving = torch.from_numpy(nib.as_closest_canonical(moving_nii).get_fdata()).float().to(device)
                static = torch.from_numpy(nib.as_closest_canonical(static_nii).get_fdata()).float().to(device)

                moving_normed = (moving - moving.min()) / (moving.max() - moving.min() + 1e-8)
                static_normed = (static - static.min()) / (static.max() - static.min() + 1e-8)

                best_loss = float('inf')
                for params in param_sampler:
                    print(f"\n\033[1mUsing {device.upper()} -- Registering CBCT: {cbct_path} with MRI: {mri_path}")
                    print(f"Testing parameters: {params}\033[0m")

                    nmi_loss_function_rigid = NMI(intensity_range=None, nbins=32, sigma=params['sigma'], use_mask=False)
                    reg_rigid = AffineRegistration(scales=(4, 2), iterations=(100, 30), is_3d=True, 
                                                    learning_rate=params['learning_rate'],
                                                    dissimilarity_function=nmi_loss_function_rigid,
                                                    optimizer=torch.optim.Adam,
                                                    with_translation=True, with_rotation=True, 
                                                    with_zoom=False, with_shear=False,
                                                    interp_mode="trilinear", padding_mode='zeros')
                    moved_image = reg_rigid(moving_normed[None, None], static_normed[None, None])[0,0]
                    loss = nmi_loss_function_rigid(moved_image[None, None], static_normed[None, None])

                    if loss < best_loss:
                        best_loss = loss
                        print(f"New best parameters found with loss: {best_loss}")
                        save_as_nifti(moved_image, cbct_path, output_path)

            else: 
                print(f"CBCT file {cbct_file} does not match the expected format: {patient_id}_CBCT_xx.nii.gz")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Register CBCT images with corresponding MRI images.')
    parser.add_argument('--cbct_folder', type=str, help='Path to the folder containing CBCT images')
    parser.add_argument('--mri_folder', type=str, help='Path to the folder containing MRI images')
    parser.add_argument('--output_folder', type=str, help='Path to the folder where output transforms will be saved')
    args = parser.parse_args()

    main(args.cbct_folder, args.mri_folder, args.output_folder)