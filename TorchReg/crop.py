import os
import json
import torch
import argparse
import numpy as np
import nibabel as nib
import SimpleITK as sitk
import torch.nn.functional as F
from torchreg.metrics import NCC
from sklearn.model_selection import ParameterSampler
from utils import normalize_image, get_corresponding_file, apply_transform_to_image
from torchreg import AffineRegistration
from metrics import NMI

def save_sitk_transform(transform_matrix, output_path, invert=False):
    """
    Save the transformation matrix as an ITK Transform file (.tfm or .h5).

    Args:
        transform_matrix (torch.Tensor): The transformation matrix from registration.
        output_path (str): Path to save the transform file.
        invert (bool): Whether to invert the transformation matrix before saving.
    """
    transform_np = transform_matrix.squeeze(0).cpu().numpy()

    if transform_np.shape[0] == 3 and transform_np.shape[1] == 4:
        transform_np = np.vstack([transform_np, [0, 0, 0, 1]])
    else:
        raise ValueError(f"Invalid transformation matrix shape: {transform_np.shape}")
    
    if invert:
        try:
            M = transform_np[:3, :3]  # Rotation/Scaling matrix
            b = transform_np[:3, 3]   # Translation vector
            M_inv = np.linalg.inv(M)  # Inverse of the rotation/scaling matrix
            b_inv = -np.dot(M_inv, b)  # Adjusted translation vector

            transform_np[:3, :3] = M_inv
            transform_np[:3, 3] = b_inv
            print(transform_np)
        except np.linalg.LinAlgError:
            raise ValueError("Transformation matrix is not invertible.")

    ndim = transform_np.shape[0] - 1

    sitk_transform = sitk.AffineTransform(ndim)
    sitk_transform.SetMatrix(transform_np[:ndim, :ndim].flatten())  # Rotation/Scaling matrix
    sitk_transform.SetTranslation(transform_np[:ndim, -1])  # Translation vector
    sitk.WriteTransform(sitk_transform, output_path)
    
def crop_volume(cbct, mri, transform_path, roi_path, patient_id, crop_cbct_path, crop_mri_path):
    """
    Crop the CBCT and MRI volumes based on the transformed ROI, normalize, and save the cropped volumes.

    Args:
        cbct (nib.Nifti1Image): The input CBCT to crop.
        mri (nib.Nifti1Image): The input MRI to crop.
        transform_path (str): Path to the transformation matrix file.
        roi_path (str): Path to the ROI .mrk.json file.
        patient_id (str): Patient ID for logging.
        crop_cbct_path (str): Path to save the cropped CBCT.
        crop_mri_path (str): Path to save the cropped MRI.
    """
    transform = sitk.ReadTransform(transform_path)
    transform_matrix = np.array(transform.GetMatrix()).reshape(3, 3)
    transform_translation = np.array(transform.GetTranslation())

    with open(roi_path, 'r') as f:
        roi_data = json.load(f)
    roi_center = np.array(roi_data['markups'][0]['center'])
    roi_size = np.array(roi_data['markups'][0]['size'])
    transformed_center = np.dot(transform_matrix, roi_center) + transform_translation

    # --- Crop CBCT volume ---
    cbct_voxel_size = cbct.header.get_zooms()[:3]  # Get voxel dimensions
    cbct_roi_size_voxels = roi_size / cbct_voxel_size  # Convert size from mm to voxels
    cbct_start_voxels = np.round((transformed_center - cbct_roi_size_voxels / 2) / cbct_voxel_size).astype(int)
    cbct_end_voxels = np.round((transformed_center + cbct_roi_size_voxels / 2) / cbct_voxel_size).astype(int)

    cbct_img_shape = cbct.shape
    cbct_start_voxels = np.clip(cbct_start_voxels, 0, np.array(cbct_img_shape) - 1)
    cbct_end_voxels = np.clip(cbct_end_voxels, 0, np.array(cbct_img_shape) - 1)

    cbct_data = cbct.get_fdata()
    cropped_cbct_data = cbct_data[cbct_start_voxels[0]:cbct_end_voxels[0],
                                  cbct_start_voxels[1]:cbct_end_voxels[1],
                                  cbct_start_voxels[2]:cbct_end_voxels[2]]
    cropped_cbct_data = (cropped_cbct_data - cropped_cbct_data.min()) / (cropped_cbct_data.max() - cropped_cbct_data.min() + 1e-8)
    cropped_image = nib.Nifti1Image(cropped_cbct_data, cbct.affine, cbct.header)
    nib.save(cropped_image, crop_cbct_path)
    print(f"Normalized and cropped volume for patient {patient_id} saved to {crop_cbct_path}")
    
    # --- Crop MRI volume ---
    mri_voxel_size = mri.header.get_zooms()[:3]  # Get voxel dimensions
    mri_roi_size_voxels = roi_size / mri_voxel_size  # Convert size from mm to voxels
    mri_start_voxels = np.round((transformed_center - mri_roi_size_voxels / 2) / mri_voxel_size).astype(int)
    mri_end_voxels = np.round((transformed_center + mri_roi_size_voxels / 2) / mri_voxel_size).astype(int)

    mri_img_shape = mri.shape
    mri_start_voxels = np.clip(mri_start_voxels, 0, np.array(mri_img_shape) - 1)
    mri_end_voxels = np.clip(mri_end_voxels, 0, np.array(mri_img_shape) - 1)

    mri_data = mri.get_fdata()
    cropped_mri_data = mri_data[mri_start_voxels[0]:mri_end_voxels[0],
                                mri_start_voxels[1]:mri_end_voxels[1],
                                mri_start_voxels[2]:mri_end_voxels[2]]
    cropped_mri_data = (cropped_mri_data - cropped_mri_data.min()) / (cropped_mri_data.max() - cropped_mri_data.min() + 1e-8)
    cropped_mri_image = nib.Nifti1Image(cropped_mri_data, mri.affine, mri.header)
    nib.save(cropped_mri_image, crop_mri_path)
    print(f"Normalized and cropped MRI for patient {patient_id} saved to {crop_mri_path}")

def main(cbct_folder, mri_folder, mean_path, roi_path, output_folder):
    """
    Main function to save the transformation matrixes between the CBCTs and the mean

    Args:
        cbct_folder (str): Path to the CBCT folder
        mri_folder (str): Path to the MRI folder
        mean_path (str): Path to the mean file
        roi_path (str): Path to the ROI mask in .mrk.json format
        output_folder (str): Path to the output folder
    """
    transform_folder = os.path.join(output_folder, "matrixes")
    crop_folder = os.path.join(output_folder, "crop")
    reg_folder = os.path.join(output_folder, "reg")
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(transform_folder, exist_ok=True)
    os.makedirs(crop_folder, exist_ok=True)
    os.makedirs(reg_folder, exist_ok=True)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    param_grid = {'learning_rate': 5*np.logspace(-4, -2, 10)}
    param_sampler = ParameterSampler(param_grid, n_iter=10)
                        
    for root, _, files in os.walk(cbct_folder):
        for cbct_file in files:
            if "_CBCT_" in cbct_file and (cbct_file.endswith(".nii") or cbct_file.endswith(".nii.gz")):
                patient_id = cbct_file.split("_CBCT_")[0]
                cbct_path = os.path.join(root, cbct_file)
                mri_path = get_corresponding_file(mri_folder, patient_id, "_MR_")

                if mean_path and mri_path:
                    transform_path = os.path.join(transform_folder, f'{patient_id}_CBCT_transform.tfm')
                    crop_cbct_path = os.path.join(crop_folder, f'{patient_id}_CBCT_crop.nii.gz')
                    crop_mri_path = os.path.join(crop_folder, f'{patient_id}_MRI_crop.nii.gz')
                    reg_path = os.path.join(reg_folder, f'{patient_id}_CBCT_registered.nii.gz')
                    
                    moving = nib.load(cbct_path)
                    static = nib.load(mean_path)

                    moving_normed = normalize_image(moving)
                    static_normed = normalize_image(static)

                    moving_tensor = torch.from_numpy(moving_normed.get_fdata()).float().to(device)
                    static_tensor = torch.from_numpy(static_normed.get_fdata()).float().to(device)
                    
                    best_loss = 0.0
                    for param in param_sampler:
                        print(f"\nUsing {device.upper()} -- Registering CBCT: {cbct_path} with mean: {mean_path}")
                        nmi_loss_function = NMI(intensity_range=None, nbins=32, sigma=0.1, use_mask=False)
                        reg_rigid = AffineRegistration(scales=(8, 4), iterations=(1000, 500), is_3d=True, 
                                                       learning_rate=param["learning_rate"], verbose=True, 
                                                       dissimilarity_function=nmi_loss_function, optimizer=torch.optim.Adam, 
                                                       with_translation=True, with_rotation=True, with_zoom=True, with_shear=False, 
                                                       align_corners=True, interp_mode="trilinear", padding_mode='zeros')

                        moved_image = reg_rigid(moving_tensor[None, None], static_tensor[None, None])
                        transform_matrix = reg_rigid.get_affine(with_grad=False)
                        
                        loss = nmi_loss_function(moved_image, static_tensor[None, None])
                        print(f"Loss (NMI): {loss}")
                        
                        if loss < best_loss:
                            best_loss = loss
                            apply_transform_to_image(cbct_path, transform_matrix, reg_path)
                            save_sitk_transform(transform_matrix, transform_path)
                            print(f"Transformation matrix saved to {transform_path}")
                           
                    mri = nib.load(mri_path)
                    crop_volume(moving, mri, transform_path, roi_path, patient_id, crop_cbct_path, crop_mri_path)                    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Register CBCT images with the mean CBCT.')
    parser.add_argument('--cbct_folder', type=str, help='Path to the folder containing CBCT images')
    parser.add_argument('--mri_folder', type=str, help='Path to the folder containing MRI images')
    parser.add_argument('--mean_path', type=str, help='Path to the folder containing the mean CBCT')
    parser.add_argument('--roi_path', type=str, help='Path to the ROI mask in .mrk.json format')
    parser.add_argument('--output_folder', type=str, help='Path to the folder where output transforms will be saved')
    
    args = parser.parse_args()
    main(args.cbct_folder, args.mri_folder, args.mean_path, args.roi_path, args.output_folder) 