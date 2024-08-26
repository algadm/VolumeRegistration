import argparse
import os
import torch
import nibabel as nib
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
from TorchIR.torchir.networks.globalnet import RigidIRNet

class MRICBCTDataset(Dataset):
    def __init__(self, cbct_folder, mri_folder):
        self.cbct_images = sorted([os.path.join(cbct_folder, f) for f in os.listdir(cbct_folder) if f.endswith('.nii.gz')])
        self.mri_images = sorted([os.path.join(mri_folder, f) for f in os.listdir(mri_folder) if f.endswith('.nii.gz')])

    def __len__(self):
        return len(self.cbct_images)

    def __getitem__(self, idx):
        cbct_img = nib.load(self.cbct_images[idx]).get_fdata()
        mri_img = nib.load(self.mri_images[idx]).get_fdata()
        print("cbct shape:", cbct_img.shape)
        print("mri shape:", mri_img.shape)
        return torch.tensor(mri_img, dtype=torch.float32).unsqueeze(0), torch.tensor(cbct_img, dtype=torch.float32).unsqueeze(0)

def main(training_cbct_folder, training_mri_folder, validation_mri_folder, testing_mri_folder, output_folder):
    # Initialize datasets and dataloaders
    # training_dataset = MRICBCTDataset(training_cbct_folder, training_mri_folder)
    # training_loader = DataLoader(training_dataset, batch_size=1, shuffle=True)

    # # Define the model for rigid registration
    # model = RigidIRNet(ndim=3)  # Make sure ndim=3 for 3D data

    # # Define optimizer and loss function
    # optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    # criterion = nn.MSELoss()  # Replace with an appropriate loss function if needed

    # # Training loop
    # for epoch in range(10):  # Adjust the number of epochs as needed
    #     model.train()
    #     for mri_img, cbct_img in training_loader:
    #         optimizer.zero_grad()
    #         Tmat, translation = model(mri_img, cbct_img)
    #         # You might need to implement a custom loss here depending on what you want to achieve.
    #         loss = criterion(Tmat, cbct_img)
    #         loss.backward()
    #         optimizer.step()
    #         print(f"Epoch {epoch + 1}, Loss: {loss.item()}")

    # # Save the model for inference
    # model_save_path = os.path.join(output_folder, 'mri_to_cbct_model.pth')
    # torch.save(model.state_dict(), model_save_path)

    # # Inference on validation and testing sets
    # inference_datasets = {
    #     "validation": MRICBCTDataset(validation_mri_folder, validation_mri_folder),
    #     "testing": MRICBCTDataset(testing_mri_folder, testing_mri_folder)
    # }

    # for dataset_name, dataset in inference_datasets.items():
    #     inference_loader = DataLoader(dataset, batch_size=1, shuffle=False)
    #     output_dataset_folder = os.path.join(output_folder, dataset_name)
    #     os.makedirs(output_dataset_folder, exist_ok=True)

    #     model.eval()
    #     with torch.no_grad():
    #         for i, (mri_img, cbct_img) in enumerate(inference_loader):
    #             Tmat, translation = model(mri_img, cbct_img)
    #             # Assuming Tmat is the transformation matrix to apply to MRI image
    #             # You will need to apply this transformation to the MRI image here.
                
    #             # For demonstration purposes, let's assume we just save the MRI image.
    #             # In practice, you would apply the transformation matrix to the MRI image.
    #             predicted_img_np = mri_img.squeeze().numpy()  # This would be the transformed MRI image

    #             # Save the predicted image
    #             output_path = os.path.join(output_dataset_folder, f"aligned_{i+1}.nii.gz")
    #             nib.save(nib.Nifti1Image(predicted_img_np, None), output_path)
    #             print(f"Saved registered image to {output_path}")
    
    model = RigidIRNet(ndim=3)
    sample_mri = torch.randn(1, 64, 64, 64)  # Example 3D tensor, adjust size as needed
    sample_cbct = torch.randn(1, 64, 64, 64)  # Example 3D tensor, adjust size as needed

    Tmat, translation = model(sample_mri, sample_cbct)
    print("Tmat shape:", Tmat.shape)
    print("Translation shape:", translation.shape)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Register CBCT images with corresponding MRI images.')
    parser.add_argument('--training_cbct_folder', type=str, help='Path to the folder containing training CBCT images')
    parser.add_argument('--training_mri_folder', type=str, help='Path to the folder containing training MRI images')
    parser.add_argument('--validation_mri_folder', type=str, help='Path to the folder containing validation MRI images')
    parser.add_argument('--testing_mri_folder', type=str, help='Path to the folder containing testing MRI images')
    parser.add_argument('--output_folder', type=str, help='Path to the folder where output transforms will be saved')
    
    args = parser.parse_args()
    main(args.training_cbct_folder, args.training_mri_folder, args.validation_mri_folder, args.testing_mri_folder, args.output_folder)







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

