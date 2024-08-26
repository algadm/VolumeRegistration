import os
import shutil
import argparse
from sklearn.model_selection import train_test_split

def split_data(cbct_dir, mri_dir, val_dir, output_dir):
    train_cbct_dir = os.path.join(output_dir, 'training/CBCT')
    train_mri_dir = os.path.join(output_dir, 'training/MRI')
    val_cbct_dir = os.path.join(output_dir, 'validation/CBCT')
    val_mri_dir = os.path.join(output_dir, 'validation/MRI')
    test_cbct_dir = os.path.join(output_dir, 'testing/CBCT')
    test_mri_dir = os.path.join(output_dir, 'testing/MRI')

    # Create directories if they don't exist
    os.makedirs(train_cbct_dir, exist_ok=True)
    os.makedirs(train_mri_dir, exist_ok=True)
    os.makedirs(val_cbct_dir, exist_ok=True)
    os.makedirs(val_mri_dir, exist_ok=True)
    os.makedirs(test_cbct_dir, exist_ok=True)
    os.makedirs(test_mri_dir, exist_ok=True)

    # Get the list of MRI files and corresponding CBCT files
    mri_files = sorted([f for f in os.listdir(mri_dir) if f.endswith('.nii.gz') and '_MR_' in f])
    cbct_files = sorted([f for f in os.listdir(cbct_dir) if f.endswith('.nii.gz') and '_CBCT_' in f])

    # Extract the patient IDs (e.g., A001) from the filenames
    mri_ids = [f.split('_MR_')[0] for f in mri_files]
    cbct_ids = [f.split('_CBCT_')[0] for f in cbct_files]

    # Ensure that every MRI has a corresponding CBCT file
    assert set(mri_ids) == set(cbct_ids), "Mismatch in the patient IDs between CBCT and MRI files."

    # Split the patient IDs into training (80%), validation (10%), and testing (10%)
    train_ids, test_val_ids = train_test_split(mri_ids, test_size=0.2, random_state=42)
    val_ids, test_ids = train_test_split(test_val_ids, test_size=0.5, random_state=42)

    # Copy the training files
    for patient_id in train_ids:
        cbct_file = next((f for f in cbct_files if patient_id in f), None)
        mri_file = next((f for f in mri_files if patient_id in f), None)
        
        if cbct_file and mri_file:
            shutil.copy2(os.path.join(cbct_dir, cbct_file), os.path.join(train_cbct_dir, cbct_file))
            shutil.copy2(os.path.join(mri_dir, mri_file), os.path.join(train_mri_dir, mri_file))

    # Copy the validation files
    for patient_id in val_ids:
        cbct_file = next((f for f in cbct_files if patient_id in f), None)
        mri_file = next((f for f in mri_files if patient_id in f), None)
        
        if cbct_file and mri_file:
            shutil.copy2(os.path.join(cbct_dir, cbct_file), os.path.join(val_cbct_dir, cbct_file))
            shutil.copy2(os.path.join(mri_dir, mri_file), os.path.join(val_mri_dir, mri_file))

            # Assuming corresponding val files have the same name structure
            val_file_src = os.path.join(val_dir, mri_file.replace('_MR_', '_VAL_'))
            val_file_dst = os.path.join(val_mri_dir, mri_file.replace('_MR_', '_VAL_'))
            if os.path.exists(val_file_src):
                shutil.copy2(val_file_src, val_file_dst)

    # Copy the testing files
    for patient_id in test_ids:
        cbct_file = next((f for f in cbct_files if patient_id in f), None)
        mri_file = next((f for f in mri_files if patient_id in f), None)
        
        if cbct_file and mri_file:
            shutil.copy2(os.path.join(cbct_dir, cbct_file), os.path.join(test_cbct_dir, cbct_file))
            shutil.copy2(os.path.join(mri_dir, mri_file), os.path.join(test_mri_dir, mri_file))

            # Assuming corresponding val files have the same name structure
            val_file_src = os.path.join(val_dir, mri_file.replace('_MR_', '_VAL_'))
            val_file_dst = os.path.join(test_mri_dir, mri_file.replace('_MR_', '_VAL_'))
            if os.path.exists(val_file_src):
                shutil.copy2(val_file_src, val_file_dst)

    print("Data split into training, validation, and testing complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Split CBCT and MRI images into training, validation, and testing sets.')
    parser.add_argument('--cbct_folder', type=str, help='Path to the folder containing CBCT images')
    parser.add_argument('--mri_folder', type=str, help='Path to the folder containing MRI images')
    parser.add_argument('--val_folder', type=str, help='Path to the folder containing the values')
    parser.add_argument('--output_folder', type=str, help='Path to the folder where output datasets will be saved')
    
    args = parser.parse_args()
    split_data(args.cbct_folder, args.mri_folder, args.val_folder, args.output_folder)
