import os
import itk
import argparse
import SimpleITK as sitk
import numpy as np
import gzip
import shutil
import subprocess
import matplotlib.pyplot as plt

def get_corresponding_file(folder, patient_id, modality):
    """Get the corresponding file for a given patient ID and modality."""
    for root, _, files in os.walk(folder):
        for file in files:
            if file.startswith(patient_id) and modality in file and file.endswith(".nii.gz"):
                return os.path.join(root, file)
    return None

def MatrixRetrieval(TransformParameterMapObject):
    """Retrieve the matrix from the transform parameter map"""
    ParameterMap = TransformParameterMapObject.GetParameterMap(0)

    if ParameterMap["Transform"][0] == "AffineTransform":
        matrix = [float(i) for i in ParameterMap["TransformParameters"]]
        # Convert to a sitk transform
        transform = sitk.AffineTransform(3)
        transform.SetParameters(matrix)

    elif ParameterMap["Transform"][0] == "EulerTransform":
        A = [float(i) for i in ParameterMap["TransformParameters"][0:3]]
        B = [float(i) for i in ParameterMap["TransformParameters"][3:6]]
        # Convert to a sitk transform
        transform = sitk.Euler3DTransform()
        transform.SetRotation(angleX=A[0], angleY=A[1], angleZ=A[2])
        transform.SetTranslation(B)

    return transform

def ComputeFinalMatrix(Transforms):
    """Compute the final matrix from the list of matrices and translations."""
    Rotation, Translation = [], []
    for i in range(len(Transforms)):
        Rotation.append(Transforms[i].GetMatrix())
        Translation.append(Transforms[i].GetTranslation())

    # Compute the final rotation matrix
    final_rotation = np.reshape(np.asarray(Rotation[0]), (3, 3))
    for i in range(1, len(Rotation)):
        final_rotation = final_rotation @ np.reshape(np.asarray(Rotation[i]), (3, 3))

    # Compute the final translation matrix
    final_translation = np.reshape(np.asarray(Translation[0]), (1, 3))
    for i in range(1, len(Translation)):
        final_translation = final_translation + np.reshape(
            np.asarray(Translation[i]), (1, 3)
        )

    # Create the final transform
    final_transform = sitk.Euler3DTransform()
    final_transform.SetCenter([0.0, 0.0, 0.0])
    final_transform.SetMatrix(final_rotation.flatten().tolist())
    final_transform.SetTranslation(final_translation[0].tolist())

    return final_transform

def ElastixReg(fixed_image_path, moving_image_path, output_directory, parameter_file_path, log_file_path=None):
    # Ensure the output directory exists
    os.makedirs(output_directory, exist_ok=True)

    # Command to run Elastix from the command line
    command = [
        "elastix",
        "-f", fixed_image_path,
        "-m", moving_image_path,
        "-out", output_directory,
        "-p", parameter_file_path
    ]

    # Run the command, redirecting stdout and stderr to the log file or suppressing them
    if log_file_path:
        print("log_path : ", log_file_path)
        with open(log_file_path, 'w') as log_file:
            subprocess.run(command, stdout=log_file, stderr=log_file, check=True)
    else:
        with open(os.devnull, 'w') as devnull:
            subprocess.run(command, stdout=devnull, stderr=devnull, check=True)

    # The resulting transformation file will typically be stored as `TransformParameters.0.txt` in the output directory
    transform_file_path = os.path.join(output_directory, "TransformParameters.0.txt")
    
    return transform_file_path

def extract_metric_from_log(log_file_path):
    with open(log_file_path, 'r') as log_file:
        lines = log_file.readlines()
        for line in lines:
            if "Final metric value" in line:
                # Extract the numeric part after the equals sign and convert to float
                metric_value_str = line.split("=")[-1].strip()
                try:
                    metric_value = float(metric_value_str)
                    return metric_value
                except ValueError:
                    print(f"Could not convert metric value to float: {metric_value_str}")
                    return float('inf')
    return float('inf')  # Return a large number if not found

def compute_custom_metric(fixed_image, registered_image):
    # For example, using Mean Squared Error (MSE)
    mse_metric = sitk.MeanSquaresImageToImageMetric()
    mse_value = mse_metric.Execute(fixed_image, registered_image)
    return mse_value

def MultiStartElastixReg(fixed_image_path, moving_image_path, output_directory, parameter_file_path, log_file_path=None, num_starts=5):
    best_transform = None
    best_metric_value = float('inf')
    best_output_dir = None

    fixed_image = sitk.ReadImage(fixed_image_path)

    for i in range(num_starts):
        # Create a unique output directory for each start
        current_output_dir = os.path.join(output_directory, f'start_{i}')
        os.makedirs(current_output_dir, exist_ok=True)

        # Modify the parameter file or initial transformation slightly here if needed
        initial_transform = sitk.Euler3DTransform()
        initial_transform.SetRotation(
            angleX=np.random.uniform(-0.05, 0.05),
            angleY=np.random.uniform(-0.05, 0.05),
            angleZ=np.random.uniform(-0.05, 0.05)
        )
        initial_transform.SetTranslation(
            [np.random.uniform(-2, 2), np.random.uniform(-2, 2), np.random.uniform(-2, 2)]
        )

        # Apply this transform as a pre-alignment step (optional)
        moving_image = sitk.ReadImage(moving_image_path)
        moving_image_resampled = sitk.Resample(moving_image, initial_transform)

        # Save the pre-aligned image (if desired)
        prealigned_image_path = os.path.join(current_output_dir, 'prealigned_moving_image.nii.gz')
        sitk.WriteImage(moving_image_resampled, prealigned_image_path)

        # Run the registration with this pre-aligned image
        log_file_path = os.path.join(current_output_dir, 'registration.log')
        transform_file_path = ElastixReg(fixed_image_path, prealigned_image_path, current_output_dir, parameter_file_path, log_file_path=log_file_path)

        # Check if the result file exists (.mhd in this case)
        result_image_path = os.path.join(current_output_dir, 'result.0.mhd')
        if not os.path.exists(result_image_path):
            print(f"Warning: Result image {result_image_path} not found. Skipping this iteration.")
            continue

        # Load the registered image from the .mhd file
        registered_image = sitk.ReadImage(result_image_path)

        # Optionally, compute a custom metric (e.g., MSE) using the registered image
        current_metric_value = compute_custom_metric(fixed_image, registered_image)

        # Determine if this is the best registration so far
        if current_metric_value < best_metric_value:
            best_metric_value = current_metric_value
            best_transform = sitk.ReadTransform(transform_file_path)
            best_output_dir = current_output_dir

    # Save the best transform if one was found
    if best_transform is not None:
        output_transform_path = os.path.join(output_directory, 'best_registration_transform.tfm')
        sitk.WriteTransform(best_transform, output_transform_path)
        print(f"Best registration result is saved in: {best_output_dir}")
        print(f"Best transform is saved in: {output_transform_path}")
    else:
        print("No successful registration was found.")

def main(cbct_folder, mri_folder, output_folder, num_starts=5):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        
    for root, _, files in os.walk(cbct_folder):
        for cbct_file in files:
            if "_CBCT_" in cbct_file and cbct_file.endswith(".nii.gz"):
                patient_id = cbct_file.split("_CBCT_")[0]
                cbct_path = os.path.join(root, cbct_file)
                mri_path = get_corresponding_file(mri_folder, patient_id, "_MR_")

                if mri_path:
                    patient_output_dir = os.path.join(output_folder, patient_id)
                    if not os.path.exists(patient_output_dir):
                        os.makedirs(patient_output_dir)

                    print(f"Registering CBCT: {cbct_path} with MRI: {mri_path}")

                    # Call the multi-start registration function
                    MultiStartElastixReg(cbct_path, mri_path, patient_output_dir, "./NMI/param2.txt", num_starts=num_starts)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Register CBCT images with corresponding MRI images.')
    parser.add_argument('--cbct_folder', type=str, help='Path to the folder containing CBCT images')
    parser.add_argument('--mri_folder', type=str, help='Path to the folder containing MRI images')
    parser.add_argument('--output_folder', type=str, help='Path to the folder where output transforms will be saved')
    
    args = parser.parse_args()
    main(args.cbct_folder, args.mri_folder, args.output_folder)