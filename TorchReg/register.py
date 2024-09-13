import os
import sys
import itk
import torch
import argparse
import numpy as np
import nibabel as nib
from metrics import NMI
from nmi import mutual_information
from torchreg import AffineRegistration

# def nmi_loss_function(moving, static):
#     return -mutual_information(moving, static, num_bins=64, normalized=True)

def get_corresponding_file(folder, patient_id, modality):
    """Get the corresponding file for a given patient ID and modality."""
    for root, _, files in os.walk(folder):
        for file in files:
            if file.startswith(patient_id) and modality in file and (file.endswith(".nii.gz") or file.endswith(".nii")):
                return os.path.join(root, file)
    return None

def save_as_nifti(moving_path, static_path, output_path):
    # Load the reference nifti file to get the affine and header information
    static_nifti = nib.load(static_path)
    
    # Create a new Nifti1Image using the numpy data and the affine from the reference image
    new_nifti = nib.Nifti1Image(moving_path.cpu().numpy(), static_nifti.affine, static_nifti.header)
    
    # Save the new NIfTI image to disk
    print("Saved registered image to:", output_path)
    nib.save(new_nifti, output_path)

def main(cbct_folder, mri_folder, output_folder):
    # Generate output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        
    for root, _, files in os.walk(cbct_folder):
        for cbct_file in files:
            if "_CBCT_" in cbct_file and (cbct_file.endswith(".nii") or cbct_file.endswith(".nii.gz")):
                patient_id = cbct_file.split("_CBCT_")[0]
                cbct_path = os.path.join(root, cbct_file)
                mri_path = get_corresponding_file(mri_folder, patient_id, "_MR_")

                if mri_path:

                    # Load images using nibabel
                    moving_nii = nib.load(mri_path)
                    static_nii = nib.load(cbct_path)
                    

                    # Get numpy arrays out of niftis
                    moving = nib.as_closest_canonical(moving_nii).get_fdata()
                    static = nib.as_closest_canonical(static_nii).get_fdata()

                    # Check if GPU is available
                    device = 'cuda' if torch.cuda.is_available() else 'cpu'
                    print(f"\n\033[1mUsing {device.upper()} -- Registering CBCT: {cbct_path} with MRI: {mri_path}\033[0m")

                    # Convert numpy arrays to torch tensors
                    moving = torch.from_numpy(moving).float().to(device)
                    static = torch.from_numpy(static).float().to(device)

                    moving_normed = (moving - moving.min()) / (moving.max() - moving.min())
                    static_normed = (static - static.min()) / (static.max() - static.min())

                    nmi_loss_function = NMI(intensity_range=None, nbins=64, sigma=0.01, use_mask=False)

                    # Intialize AffineRegistration
                    reg = AffineRegistration(scales=(3, 1), iterations=(100, 20), is_3d=True, learning_rate=1e-3,
                                             verbose=True, dissimilarity_function=nmi_loss_function.metric, optimizer=torch.optim.Adam,
                                             init_translation=None, init_rotation=None, init_zoom=None, init_shear=None,
                                             with_translation=True, with_rotation=True, with_zoom=False, with_shear=False,
                                             align_corners=True, interp_mode="trilinear", padding_mode='zeros')

                    # Run it!
                    moved_image = reg(moving_normed[None, None],
                                      static_normed[None, None])

                    moved_image = moved_image[0, 0]
                    moved_image = moving.max() * moved_image

                    # Save the registered image as a NIfTI file
                    output_path = os.path.join(output_folder, f'{patient_id}_registered.nii.gz')
                    save_as_nifti(moved_image, cbct_path, output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Register CBCT images with corresponding MRI images.')
    parser.add_argument('--cbct_folder', type=str, help='Path to the folder containing CBCT images')
    parser.add_argument('--mri_folder', type=str, help='Path to the folder containing MRI images')
    parser.add_argument('--output_folder', type=str, help='Path to the folder where output transforms will be saved')
    
    args = parser.parse_args()
    main(args.cbct_folder, args.mri_folder, args.output_folder)