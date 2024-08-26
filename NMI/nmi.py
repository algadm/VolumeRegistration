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

def main(cbct_folder, mri_folder, output_folder):
    # Generate output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        
    for root, _, files in os.walk(cbct_folder):
        for cbct_file in files:
            if "_CBCT_" in cbct_file and cbct_file.endswith(".nii.gz"):
                patient_id = cbct_file.split("_CBCT_")[0]
                cbct_path = os.path.join(root, cbct_file)
                mri_path = get_corresponding_file(mri_folder, patient_id, "_MR_")

                if mri_path:
                    Transforms = []
                    patient_output_dir = os.path.join(output_folder, patient_id)
                    if not os.path.exists(patient_output_dir):
                        os.makedirs(patient_output_dir)

                    print(f"Registering CBCT: {cbct_path} with MRI: {mri_path}")
                    log_file_path = os.path.join(patient_output_dir, f'{patient_id}_registration.log')
                    parameter_file_path = "./NMI/param2.txt"

                    # Run Elastix registration and get the transform file path
                    transform_file_path = ElastixReg(cbct_path, mri_path, patient_output_dir, parameter_file_path, log_file_path=log_file_path)

                    # Load the transform parameter object using SimpleITK
                    transform_object = itk.ParameterObject.New()
                    transform_object.ReadParameterFile(transform_file_path)

                    # Retrieve the transformation as a matrix
                    transform = MatrixRetrieval(transform_object)
                    Transforms.append(transform)
                    
                    # Combine the transformations (if there are multiple)
                    final_transform = ComputeFinalMatrix(Transforms)

                    # Save the final combined transform in the correct format
                    output_transform_path = os.path.join(patient_output_dir, f'{patient_id}_reg.tfm')
                    sitk.WriteTransform(final_transform, output_transform_path)

                    print("Output transform path:", output_transform_path)
                    print(f"Saved transform to {output_transform_path}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Register CBCT images with corresponding MRI images.')
    parser.add_argument('--cbct_folder', type=str, help='Path to the folder containing CBCT images')
    parser.add_argument('--mri_folder', type=str, help='Path to the folder containing MRI images')
    parser.add_argument('--output_folder', type=str, help='Path to the folder where output transforms will be saved')
    
    args = parser.parse_args()
    main(args.cbct_folder, args.mri_folder, args.output_folder)