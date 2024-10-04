import os
import torch
import argparse
import numpy as np
import nibabel as nib
from metrics import NMI
from torchreg import AffineRegistration
from sklearn.model_selection import ParameterSampler

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

def main(cbct_folder, mri_folder, output_folder, param_sampler):
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
                    best_loss = float('inf')  # Initialize to track the best loss
                    best_params = None        # Initialize to store the best parameter combination
                    # Iterate over the parameter samples
                    for params in param_sampler:
                        # Load images using nibabel
                        moving_nii = nib.load(mri_path)
                        static_nii = nib.load(cbct_path)

                        # Get numpy arrays out of niftis
                        moving = nib.as_closest_canonical(moving_nii).get_fdata()
                        static = nib.as_closest_canonical(static_nii).get_fdata()

                        # Check if GPU is available
                        device = 'cuda' if torch.cuda.is_available() else 'cpu'
                        print(f"\n\033[1mUsing {device.upper()} -- Registering CBCT: {cbct_path} with MRI: {mri_path}")

                        # Convert numpy arrays to torch tensors
                        moving = torch.from_numpy(moving).float().to(device)
                        static = torch.from_numpy(static).float().to(device)

                        # Normalize the images
                        epsilon = 1e-8    # Small value to avoid division by zero
                        moving_normed = (moving - moving.min()) / (moving.max() - moving.min() + epsilon)
                        static_normed = (static - static.min()) / (static.max() - static.min() + epsilon)

                        # Print the current parameter values
                        print(f"Testing parameters: {params}\033[0m")

                        # Initialize NMI loss function for rigid registration
                        nmi_loss_function_rigid = NMI(intensity_range=None, nbins=params['nbins_rigid'], sigma=params['sigma_rigid'], use_mask=False)

                        # Initialize AffineRegistration for Rigid registration
                        reg_rigid = AffineRegistration(scales=(4, 2), iterations=(100, 30), is_3d=True, 
                                                       learning_rate=params['learning_rate_rigid'],
                                                       verbose=True, dissimilarity_function=nmi_loss_function_rigid.metric,
                                                       optimizer=torch.optim.Adam, with_translation=True, with_rotation=True, 
                                                       with_zoom=False, with_shear=False, align_corners=True,
                                                       interp_mode="trilinear", padding_mode='zeros')

                        # Perform rigid registration
                        moved_image = reg_rigid(moving_normed[None, None],
                                                static_normed[None, None])
                        
                        moved_image = moved_image[0, 0]

                        # Calculate the final loss
                        final_loss = -nmi_loss_function_rigid.metric(moved_image[None, None], static_normed[None, None])
                        print(f"Final Loss (NMI): {final_loss}")

                        moved_image = moving.max() * moved_image

                        # Check if this is the best loss so far
                        if final_loss < best_loss and final_loss > 1e-5:
                            best_loss = final_loss
                            best_params = params
                            print(f"New best parameters found with loss: {best_loss}")

                            # Save the registered image as a NIfTI file
                            output_path = os.path.join(output_folder, f'{patient_id}_MR_registered.nii.gz')
                            save_as_nifti(moved_image, cbct_path, output_path)

                    # Print the best result at the end
                    print(f"Best parameters: {best_params}")
                    print(f"Best NMI loss: {best_loss}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Register CBCT images with corresponding MRI images.')
    parser.add_argument('--cbct_folder', type=str, help='Path to the folder containing CBCT images')
    parser.add_argument('--mri_folder', type=str, help='Path to the folder containing MRI images')
    parser.add_argument('--output_folder', type=str, help='Path to the folder where output transforms will be saved')
    
    args = parser.parse_args()

    # Define the parameter grid for hyperparameter search
    param_grid = {
        'learning_rate_rigid': np.logspace(-5, -3, 15),   # Learning rate for rigid registration
        'nbins_rigid': [64],                              # Number of bins for rigid registration NMI
        'sigma_rigid': np.logspace(-3, -2, 3)             # Sigma for rigid NMI
    }

    # Number of parameter combinations to sample
    n_samples = 45
    param_sampler = ParameterSampler(param_grid, n_iter=n_samples)

    main(args.cbct_folder, args.mri_folder, args.output_folder, param_sampler)