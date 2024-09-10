import os
import itk
import torch
import argparse
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from network import LitAIRNet  # Import your network
from torch.utils.data import DataLoader, Dataset

class MRI_CBCT_Dataset(Dataset):
    """Custom dataset for loading MRI (moving) and CBCT (fixed) image pairs."""
    def __init__(self, cbct_dir, mri_dir):
        self.cbct_files = sorted([os.path.join(cbct_dir, f) for f in os.listdir(cbct_dir) if f.endswith('.nii.gz')])
        self.mri_files = sorted([os.path.join(mri_dir, f) for f in os.listdir(mri_dir) if f.endswith('.nii.gz')])

    def __len__(self):
        return len(self.cbct_files)

    def __getitem__(self, idx):
        cbct_image = self.load_image(self.cbct_files[idx])  # Load CBCT (fixed)
        mri_image = self.load_image(self.mri_files[idx])    # Load MRI (moving)

        # Convert images to float32
        cbct_image = cbct_image.astype(np.float32)
        mri_image = mri_image.astype(np.float32)

        # Ensure the images are 3D with a single channel, i.e., (1, depth, height, width)
        if len(cbct_image.shape) == 3:  # If the image has no channel dimension
            cbct_image = np.expand_dims(cbct_image, axis=0)
        if len(mri_image.shape) == 3:  # If the image has no channel dimension
            mri_image = np.expand_dims(mri_image, axis=0)

        # Permute MRI to match CBCT (depth should be the last dimension)
        mri_image = np.transpose(mri_image, (0, 3, 2, 1))  # Change (depth, height, width) to (1, height, width, depth)

        print(f'Loaded CBCT image {idx}: {cbct_image.shape}')
        print(f'Loaded MRI image {idx}: {mri_image.shape}')

        return {'fixed': cbct_image, 'moving': mri_image}

    def load_image(self, file_path):
        """Load the NIfTI image using ITK and convert it to a NumPy array."""
        itk_image = itk.imread(file_path)
        np_image = itk.array_view_from_image(itk_image)
        return np_image

def main(training_path, validation_path, testing_path, output_path):
    # Set up paths
    training_cbct_path = os.path.join(training_path, 'CBCT')
    training_mri_path = os.path.join(training_path, 'MRI')
    validation_cbct_path = os.path.join(validation_path, 'CBCT')
    validation_mri_path = os.path.join(validation_path, 'MRI')
    testing_cbct_path = os.path.join(testing_path, 'CBCT')
    testing_mri_path = os.path.join(testing_path, 'MRI')

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Load datasets
    train_dataset = MRI_CBCT_Dataset(training_cbct_path, training_mri_path)
    val_dataset = MRI_CBCT_Dataset(validation_cbct_path, validation_mri_path)
    test_dataset = MRI_CBCT_Dataset(testing_cbct_path, testing_mri_path)

    batch_size = 4  # Adjust as needed
    train_loader = DataLoader(train_dataset, batch_size=batch_size) # , shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Initialize the model
    model = LitAIRNet()  # Make sure this network is set up for MRI to CBCT registration

    # Configure the model checkpointing
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',  # or whatever metric you want to monitor
        dirpath=output_path,  # path to save checkpoints
        filename='best_model-{epoch:02d}-{val_loss:.2f}',  # naming convention for saved models
        save_top_k=1,  # save only the best model
        mode='min'  # because we want to minimize the validation loss
    )

    # Initialize the PyTorch Lightning trainer
    trainer = pl.Trainer(default_root_dir=output_path,
                         log_every_n_steps=50,
                         val_check_interval=50,
                         max_epochs=100,
                         accelerator="gpu",
                         devices=1,
                         callbacks=[checkpoint_callback],
                         num_sanity_val_steps=batch_size)
    
    # Train the model
    trainer.fit(model, train_loader, val_loader)

    # Assuming you want to test the model as well
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    trainer.test(model, test_loader, ckpt_path='best')  # Load the best model for testing

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Register CBCT images with corresponding MRI images.')
    parser.add_argument('--training_path', type=str, required=True, help='Path to the folder containing training images')
    parser.add_argument('--validation_path', type=str, required=True, help='Path to the folder containing validation images')
    parser.add_argument('--testing_path', type=str, required=True, help='Path to the folder containing testing images')
    parser.add_argument('--output_path', type=str, required=True, help='Path to the folder where output transforms will be saved')

    args = parser.parse_args()
    main(args.training_path, args.validation_path, args.testing_path, args.output_path)
