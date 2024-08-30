import os
import sys
sys.path.append('/home/lucia/Documents/Alban/VolumeRegistration')
import torch
from TorchIR.torchir.networks.globalnet import RigidIRNet
from TorchIR.torchir.metrics import NMI
import TorchIR.torchir as tir
import SimpleITK as sitk
import torch.nn.functional as F
import numpy as np
import time

def load_image(image_path):
    return sitk.GetArrayFromImage(sitk.ReadImage(image_path))

def save_image(image_np, reference_image_path, output_path):
    reference_image = sitk.ReadImage(reference_image_path)
    image_sitk = sitk.GetImageFromArray(image_np)
    image_sitk.CopyInformation(reference_image)
    
    U, _, Vt = np.linalg.svd(np.array(image_sitk.GetDirection()).reshape(3, 3))
    orthonormal_direction = np.dot(U, Vt)
    image_sitk.SetDirection(orthonormal_direction.flatten().tolist())
    
    sitk.WriteImage(image_sitk, output_path)

def apply_transformation_with_resampling(moving_image, Tmat, translation, fixed_image_shape):
    """
    Apply the affine transformation to the moving image and resample it to the fixed image's space.
    Args:
        moving_image (torch.Tensor): The moving image tensor.
        Tmat (torch.Tensor): The rotation matrix from the model.
        translation (torch.Tensor): The translation vector from the model.
        fixed_image_shape (tuple): The shape of the fixed image.
    Returns:
        torch.Tensor: The transformed and resampled moving image.
    """
    batch_size = moving_image.shape[0]
    ndim = 3  # Since we are dealing with 3D data

    # Create a 3x4 affine transformation matrix
    affine_matrix = torch.eye(ndim + 1).unsqueeze(0).repeat(batch_size, 1, 1)  # [batch_size, 4, 4]
    affine_matrix[:, :3, :3] = Tmat
    affine_matrix[:, :3, 3] = translation.squeeze(-1)

    # Remove the last row to use with affine_grid, which expects a [3, 4] matrix
    affine_matrix = affine_matrix[:, :3, :]  # [batch_size, 3, 4] for 3D

    # Generate the grid in the fixed image's coordinate space
    grid = F.affine_grid(affine_matrix, fixed_image_shape, align_corners=False)

    # Sample the moving image according to this grid
    transformed_image = F.grid_sample(moving_image, grid, align_corners=False, mode='bilinear')

    return transformed_image

def perform_registration(fixed_image_np, moving_image_np):
    # Convert numpy arrays to torch tensors
    fixed_image_tensor = torch.from_numpy(fixed_image_np).float().unsqueeze(0).unsqueeze(0)
    moving_image_tensor = torch.from_numpy(moving_image_np).float().unsqueeze(0).unsqueeze(0)

    # Initialize the transformation model
    model = RigidIRNet(ndim=3)  # Assuming 3D data

    # Set the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Define loss function using the NMI class
    loss_fn = NMI()

    # Training loop for rigid registration
    num_epochs = 1
    num_iterations = 10
    for epoch in range(num_epochs):
        for i in range(num_iterations):
            start_time = time.time()
            optimizer.zero_grad()

            # Perform the forward pass with both fixed and moving images
            Tmat, translation = model(fixed_image_tensor, moving_image_tensor)

            # Apply the transformation and resample to fixed image space
            registered_moving_image = apply_transformation_with_resampling(moving_image_tensor, Tmat, translation, fixed_image_tensor.size())

            # Compute the loss
            loss = loss_fn(registered_moving_image, fixed_image_tensor)
            loss.backward()
            optimizer.step()
            end_time = time.time()

            if i % 1 == 0:
                print(f"Epoch {epoch}, Iteration {i}, Loss: {loss.item()}   in {end_time - start_time:.4f} seconds")
        print("\n")

    # Get the registered image as numpy array
    registered_image_np = registered_moving_image.detach().numpy().squeeze()

    return registered_image_np


def process_dataset(dataset_dir, output_dir):
    for split in ['training', 'validation', 'testing']:
        cbct_dir = os.path.join(dataset_dir, split, 'CBCT')
        mri_dir = os.path.join(dataset_dir, split, 'MRI')
        split_output_dir = os.path.join(output_dir, split)

        if not os.path.exists(split_output_dir):
            os.makedirs(split_output_dir)

        for cbct_file in os.listdir(cbct_dir):
            cbct_path = os.path.join(cbct_dir, cbct_file)
            mri_file = cbct_file.replace('_CBCT_Crop_MA', '_MR_OR_cropLeft')
            mri_path = os.path.join(mri_dir, mri_file)

            if os.path.exists(mri_path):
                print(f"Processing {cbct_file} and {mri_file}")

                # Load images
                fixed_image_np = load_image(cbct_path)
                moving_image_np = load_image(mri_path)

                # Perform registration
                registered_image_np = perform_registration(fixed_image_np, moving_image_np)

                # Save the registered image
                output_path = os.path.join(split_output_dir, f"registered_{mri_file}")
                save_image(registered_image_np, cbct_path, output_path)

                print(f"Registered image saved to {output_path}")
            else:
                print(f"Matching MRI file for {cbct_file} not found.")

def main():
    # Paths
    dataset_dir = "./DeepLearning/dataset"
    output_dir = "./DeepLearning/output"

    # Process the entire dataset
    process_dataset(dataset_dir, output_dir)

if __name__ == "__main__":
    main()
















# import argparse
# import os
# import torch
# import nibabel as nib
# from torch.utils.data import DataLoader, Dataset
# import torch.nn as nn
# from TorchIR.torchir.networks.globalnet import RigidIRNet

# class MRICBCTDataset(Dataset):
#     def __init__(self, cbct_folder, mri_folder):
#         self.cbct_images = sorted([os.path.join(cbct_folder, f) for f in os.listdir(cbct_folder) if f.endswith('.nii.gz')])
#         self.mri_images = sorted([os.path.join(mri_folder, f) for f in os.listdir(mri_folder) if f.endswith('.nii.gz')])

#     def __len__(self):
#         return len(self.cbct_images)

#     def __getitem__(self, idx):
#         cbct_img = nib.load(self.cbct_images[idx]).get_fdata()
#         mri_img = nib.load(self.mri_images[idx]).get_fdata()
#         print("cbct shape:", cbct_img.shape)
#         print("mri shape:", mri_img.shape)
#         return torch.tensor(mri_img, dtype=torch.float32).unsqueeze(0), torch.tensor(cbct_img, dtype=torch.float32).unsqueeze(0)

# def main(training_cbct_folder, training_mri_folder, validation_mri_folder, testing_mri_folder, output_folder):
#     # Initialize datasets and dataloaders
#     # training_dataset = MRICBCTDataset(training_cbct_folder, training_mri_folder)
#     # training_loader = DataLoader(training_dataset, batch_size=1, shuffle=True)

#     # # Define the model for rigid registration
#     # model = RigidIRNet(ndim=3)  # Make sure ndim=3 for 3D data

#     # # Define optimizer and loss function
#     # optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
#     # criterion = nn.MSELoss()  # Replace with an appropriate loss function if needed

#     # # Training loop
#     # for epoch in range(10):  # Adjust the number of epochs as needed
#     #     model.train()
#     #     for mri_img, cbct_img in training_loader:
#     #         optimizer.zero_grad()
#     #         Tmat, translation = model(mri_img, cbct_img)
#     #         # You might need to implement a custom loss here depending on what you want to achieve.
#     #         loss = criterion(Tmat, cbct_img)
#     #         loss.backward()
#     #         optimizer.step()
#     #         print(f"Epoch {epoch + 1}, Loss: {loss.item()}")

#     # # Save the model for inference
#     # model_save_path = os.path.join(output_folder, 'mri_to_cbct_model.pth')
#     # torch.save(model.state_dict(), model_save_path)

#     # # Inference on validation and testing sets
#     # inference_datasets = {
#     #     "validation": MRICBCTDataset(validation_mri_folder, validation_mri_folder),
#     #     "testing": MRICBCTDataset(testing_mri_folder, testing_mri_folder)
#     # }

#     # for dataset_name, dataset in inference_datasets.items():
#     #     inference_loader = DataLoader(dataset, batch_size=1, shuffle=False)
#     #     output_dataset_folder = os.path.join(output_folder, dataset_name)
#     #     os.makedirs(output_dataset_folder, exist_ok=True)

#     #     model.eval()
#     #     with torch.no_grad():
#     #         for i, (mri_img, cbct_img) in enumerate(inference_loader):
#     #             Tmat, translation = model(mri_img, cbct_img)
#     #             # Assuming Tmat is the transformation matrix to apply to MRI image
#     #             # You will need to apply this transformation to the MRI image here.
                
#     #             # For demonstration purposes, let's assume we just save the MRI image.
#     #             # In practice, you would apply the transformation matrix to the MRI image.
#     #             predicted_img_np = mri_img.squeeze().numpy()  # This would be the transformed MRI image

#     #             # Save the predicted image
#     #             output_path = os.path.join(output_dataset_folder, f"aligned_{i+1}.nii.gz")
#     #             nib.save(nib.Nifti1Image(predicted_img_np, None), output_path)
#     #             print(f"Saved registered image to {output_path}")
    
#     model = RigidIRNet(ndim=3)
#     sample_mri = torch.randn(1, 64, 64, 64)  # Example 3D tensor, adjust size as needed
#     sample_cbct = torch.randn(1, 64, 64, 64)  # Example 3D tensor, adjust size as needed

#     Tmat, translation = model(sample_mri, sample_cbct)
#     print("Tmat shape:", Tmat.shape)
#     print("Translation shape:", translation.shape)

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description='Register CBCT images with corresponding MRI images.')
#     parser.add_argument('--training_cbct_folder', type=str, help='Path to the folder containing training CBCT images')
#     parser.add_argument('--training_mri_folder', type=str, help='Path to the folder containing training MRI images')
#     parser.add_argument('--validation_mri_folder', type=str, help='Path to the folder containing validation MRI images')
#     parser.add_argument('--testing_mri_folder', type=str, help='Path to the folder containing testing MRI images')
#     parser.add_argument('--output_folder', type=str, help='Path to the folder where output transforms will be saved')
    
#     args = parser.parse_args()
#     main(args.training_cbct_folder, args.training_mri_folder, args.validation_mri_folder, args.testing_mri_folder, args.output_folder)







# import os
# import torch
# import SimpleITK as sitk
# import argparse
# import numpy as np

# # Import the necessary modules from TorchIR
# from TorchIR.torchir.dlir_framework import DLIRFramework
# from TorchIR.torchir.transformers import AffineTransformer
# from TorchIR.torchir.networks.globalnet import RigidIRNet

# def load_nifti_image(file_path):
#     """Load a NIfTI image from a file."""
#     return sitk.ReadImage(file_path)

# def save_nifti_image(image, file_path):
#     """Save a NIfTI image to a file."""
#     sitk.WriteImage(image, file_path)

# def register_mri_to_cbct(mri_array, cbct_array, device='cuda'):
#     """Perform MRI to CBCT registration."""
#     # Ensure the data is in the correct shape
#     if len(mri_array.shape) == 3:
#         # Add batch and channel dimensions for 3D images
#         mri_array = mri_array[np.newaxis, np.newaxis, :, :, :]  # [1, 1, Depth, Height, Width]
#         cbct_array = cbct_array[np.newaxis, np.newaxis, :, :, :]  # [1, 1, Depth, Height, Width]

#     mri_tensor = torch.from_numpy(mri_array).float().to(device)
#     cbct_tensor = torch.from_numpy(cbct_array).float().to(device)

#     # Set up the registration model with 3D convolutions
#     model = DLIRFramework(only_last_trainable=True)
#     transformer = AffineTransformer(ndim=3)  # 3D transformer
#     model.add_stage(RigidIRNet(in_channels=1, out_channels=32), transformer)  # Ensure RigidIRNet uses conv3d
#     model.to(device)

#     # Perform registration
#     registered_mri_tensor = model(cbct_tensor, mri_tensor)
#     return registered_mri_tensor.squeeze(0).cpu().numpy()

# def process_registration(mri_folder, cbct_folder, output_folder, device='cuda'):
#     """Process registration for all files in the folder."""
#     mri_files = sorted([f for f in os.listdir(mri_folder) if f.endswith('.nii.gz')])
#     cbct_files = sorted([f for f in os.listdir(cbct_folder) if f.endswith('.nii.gz')])

#     for mri_file, cbct_file in zip(mri_files, cbct_files):
#         # Load MRI and CBCT images
#         mri_image = load_nifti_image(os.path.join(mri_folder, mri_file))
#         cbct_image = load_nifti_image(os.path.join(cbct_folder, cbct_file))

#         # Convert to arrays
#         mri_array = sitk.GetArrayFromImage(mri_image)
#         cbct_array = sitk.GetArrayFromImage(cbct_image)

#         # Perform registration
#         registered_mri_array = register_mri_to_cbct(mri_array, cbct_array, device=device)

#         # Save the result
#         registered_mri_image = sitk.GetImageFromArray(registered_mri_array)
#         registered_mri_image.CopyInformation(mri_image)
#         output_path = os.path.join(output_folder, f'registered_{mri_file}')
#         save_nifti_image(registered_mri_image, output_path)

# def main(training_cbct_folder, training_mri_folder, validation_mri_folder, testing_mri_folder, output_folder):
#     """Main function to handle the registration of MRI to CBCT for all datasets."""
    
#     # Register training data
#     print("Registering training data...")
#     process_registration(training_mri_folder, training_cbct_folder, output_folder)

#     # Validate the model
#     print("Evaluating model on validation data...")
#     process_registration(validation_mri_folder, training_cbct_folder, output_folder)

#     # Register testing data
#     print("Registering testing data...")
#     process_registration(testing_mri_folder, training_cbct_folder, output_folder)

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description='Register CBCT images with corresponding MRI images.')
#     parser.add_argument('--training_cbct_folder', type=str, help='Path to the folder containing training CBCT images')
#     parser.add_argument('--training_mri_folder', type=str, help='Path to the folder containing training MRI images')
#     parser.add_argument('--validation_mri_folder', type=str, help='Path to the folder containing validation MRI images')
#     parser.add_argument('--testing_mri_folder', type=str, help='Path to the folder containing testing MRI images')
#     parser.add_argument('--output_folder', type=str, help='Path to the folder where output transforms will be saved')
    
#     args = parser.parse_args()
#     main(args.training_cbct_folder, args.training_mri_folder, args.validation_mri_folder, args.testing_mri_folder, args.output_folder)

