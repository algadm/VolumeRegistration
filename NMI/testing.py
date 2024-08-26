import os
import itk
import argparse
import SimpleITK as sitk
import numpy as np
import gzip
import shutil
import matplotlib.pyplot as plt

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
    """Compute the final matrix from the list of matrices and translations"""
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
    final_transform.SetMatrix(final_rotation.flatten().tolist())
    final_transform.SetTranslation(final_translation[0].tolist())

    return final_transform

def invert_image(image):
    # Ensure the image is of a supported type (e.g., float)
    image = itk.CastImageFilter[itk.Image[itk.F, 3], itk.Image[itk.F, 3]].New(Input=image)
    image.Update()
    image = image.GetOutput()

    # Compute the maximum intensity value in the image
    stats_filter = itk.MinimumMaximumImageFilter.New(Input=image)
    stats_filter.Update()
    max_intensity = stats_filter.GetMaximum()

    # Create an image filled with the maximum intensity value
    max_image = itk.Image[itk.F, 3].New()
    max_image.SetRegions(image.GetLargestPossibleRegion())
    max_image.CopyInformation(image)
    max_image.Allocate()
    max_image.FillBuffer(max_intensity)

    # Subtract each pixel's intensity from the maximum intensity
    invert_filter = itk.SubtractImageFilter.New(Input1=max_image, Input2=image)
    invert_filter.Update()
    inverted_image = invert_filter.GetOutput()

    return inverted_image

def normalize_image(image, output_min=0.0, output_max=1.0):
    """Normalize the image intensity to a specified range."""
    rescale_filter = itk.RescaleIntensityImageFilter.New(Input=image)
    rescale_filter.SetOutputMinimum(output_min)
    rescale_filter.SetOutputMaximum(output_max)
    rescale_filter.Update()
    normalized_image = rescale_filter.GetOutput()
    return normalized_image

# def apply_transform_and_save(transform_path, mri_path, output_path):
#     """Apply the .tfm transform to the image and save the result as .nii.gz"""
#     # Read the transformation
#     transform = sitk.ReadTransform(transform_path)

#     # Read the image
#     image = sitk.ReadImage(mri_path)
#     image_array = sitk.GetArrayFromImage(image)
#     print(f"Min intensity: {image_array.min()}")
#     print(f"Max intensity: {image_array.max()}")

#     # Apply the transformation
#     resampled_image = sitk.Resample(image, transform, defaultPixelValue=0.0, interpolator=sitk.sitkLinear)
#     resampled_image_array = sitk.GetArrayFromImage(resampled_image)
#     print(f"Min intensity: {resampled_image_array.min()}")
#     print(f"Max intensity: {resampled_image_array.max()}")

#     # Save the transformed image
#     sitk.WriteImage(resampled_image, output_path)
#     print(f"Saved transformed image: {output_path}")

def apply_transform_and_save(transform_path, moving_image_path, output_folder):
    # Load the image
    moving_image = sitk.ReadImage(moving_image_path)

    # Define the temporary directory for Transformix output
    temp_dir = os.path.join(output_folder, "temp_transformix_output")
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)

    # Write the transform parameter file to the temporary directory
    temp_transform_param_file = os.path.join(temp_dir, "transform_parameters.txt")
    shutil.copy(transform_path, temp_transform_param_file)

    # Create the Transformix command
    command = f'transformix -in {moving_image_path} -out {temp_dir} -tp {temp_transform_param_file}'

    # Execute Transformix
    os.system(command)

    # Read the result image
    result_image_path = os.path.join(temp_dir, "result.nii")
    result_image = sitk.ReadImage(result_image_path)

    # Save the transformed image
    sitk.WriteImage(result_image, output_folder)
    print(f"Transformed image saved to: {output_folder}")

    # Clean up temporary directory
    shutil.rmtree(temp_dir)

def ElastixReg(fixed_image, moving_image, initial_transform=None, log_file_path=None):
    """Perform a registration using elastix with Normalized Mutual Information"""
        
    fixed_image = invert_image(fixed_image)

    elastix_object = itk.ElastixRegistrationMethod.New(fixed_image, moving_image)

    # ParameterMap
    parameter_object = itk.ParameterObject.New()
    default_rigid_parameter_map = parameter_object.GetDefaultParameterMap("rigid")

    default_rigid_parameter_map["Metric"] = ["NormalizedMutualInformation"]
    default_rigid_parameter_map["Optimizer"] = ["AdaptiveStochasticGradientDescent"]
    # RegularStepGradientDescent, AdaptiveStochasticGradientDescent
    default_rigid_parameter_map["ASGDParameterEstimationMethod"] = ["DisplacementMagnitude"]
    # Original, DisplacementMagnitude, ImageDiscrepancy
    # default_rigid_parameter_map["SigmoidScaleFactor"] = ["0.1"]
    default_rigid_parameter_map["Metric0Weight"] = ["1.0"]
    default_rigid_parameter_map["NumberOfHistogramBins"] = ["64"]
    default_rigid_parameter_map["UseNormalization"] = ["false"]
    default_rigid_parameter_map["FixedImagePyramid"] = ["FixedSmoothingImagePyramid"]
    default_rigid_parameter_map["MovingImagePyramid"] = ["MovingSmoothingImagePyramid"]
    default_rigid_parameter_map["FixedImagePyramidSchedule"] = ["8", "8", "8", "4", "4", "4", "2", "2", "2", "1", "1", "1"]
    default_rigid_parameter_map["MovingImagePyramidSchedule"] = ["8", "8", "8", "4", "4", "4", "2", "2", "2", "1", "1", "1"]

    # ["8", "8", "8", "4", "4", "4", "2", "2", "2", "1", "1", "1"]
    # default_rigid_parameter_map["Scales"] = ["3000", "3000", "3000", "0.05", "0.5", "0.05"]

    default_rigid_parameter_map["AutomaticScalesEstimation"] = ["true"]
    default_rigid_parameter_map["AutomaticTransformInitialization"] = ["true"]
    default_rigid_parameter_map["AutomaticTransformInitializationMethod"] = ["GeometricalCenter"]
    # CenterOfGravity, GeometricalCenter
    
    parameter_object.AddParameterMap(default_rigid_parameter_map)
    if initial_transform is not None:
        elastix_object.SetInitialTransformParameterObject(initial_transform)

    parameter_object.SetParameter("WriteResultImage", "false")
    parameter_object.SetParameter("ComputeZYX", "true")
    parameter_object.SetParameter("MaximumStepLength", "0.01")
    parameter_object.SetParameter("MaximumNumberOfIterations", "1000")
    parameter_object.SetParameter("NumberOfResolutions", "3")
    parameter_object.SetParameter("NumberOfSpatialSamples", "10000")
    parameter_object.SetParameter("UseMultiThreadingForMetrics", "true")
    
    elastix_object.SetParameterObject(parameter_object)

    # Additional parameters
    elastix_object.SetLogToConsole(False)
    if log_file_path:
        print("log_path : ", log_file_path)
        log_directory = os.path.dirname(log_file_path)
        elastix_object.LogToFileOn()
        elastix_object.SetLogFileName(log_file_path)
        elastix_object.SetOutputDirectory(log_directory)
        print("pass log file")
    else:
        elastix_object.LogToConsoleOn()

    # Execute registration
    elastix_object.UpdateLargestPossibleRegion()

    TransParamObj = elastix_object.GetTransformParameterObject()

    return TransParamObj

def SecondElastixReg(fixed_image, moving_image, initial_transform=None, log_file_path=None):
    """Perform a registration using elastix with Normalized Mutual Information"""
        
    fixed_image = invert_image(fixed_image)

    elastix_object = itk.ElastixRegistrationMethod.New(fixed_image, moving_image)

    # ParameterMap
    parameter_object = itk.ParameterObject.New()
    default_rigid_parameter_map = parameter_object.GetDefaultParameterMap("rigid")

    default_rigid_parameter_map["Metric"] = ["NormalizedMutualInformation"]
    # NormalizedMutualInformation, AdvancedMattesMutualInformation
    default_rigid_parameter_map["Optimizer"] = ["AdaptiveStochasticGradientDescent"]
    # RegularStepGradientDescent, AdaptiveStochasticGradientDescent
    default_rigid_parameter_map["ASGDParameterEstimationMethod"] = ["ImageDiscrepancy"]
    # Original, DisplacementMagnitude, ImageDiscrepancy
    # default_rigid_parameter_map["SigmoidScaleFactor"] = ["0.1"]
    default_rigid_parameter_map["Metric0Weight"] = ["1.0"]
    default_rigid_parameter_map["NumberOfHistogramBins"] = ["64"]
    default_rigid_parameter_map["UseNormalization"] = ["false"]
    default_rigid_parameter_map["FixedImagePyramid"] = ["FixedSmoothingImagePyramid"]
    default_rigid_parameter_map["MovingImagePyramid"] = ["MovingSmoothingImagePyramid"]
    default_rigid_parameter_map["FixedImagePyramidSchedule"] = ["8", "8", "8", "4", "4", "4", "2", "2", "2", "1", "1", "1"]
    default_rigid_parameter_map["MovingImagePyramidSchedule"] = ["8", "8", "8", "4", "4", "4", "2", "2", "2", "1", "1", "1"]

    # ["8", "8", "8", "4", "4", "4", "2", "2", "2", "1", "1", "1"]
    # default_rigid_parameter_map["Scales"] = ["3000", "3000", "3000", "0.05", "0.5", "0.05"]

    default_rigid_parameter_map["AutomaticScalesEstimation"] = ["true"]
    default_rigid_parameter_map["AutomaticTransformInitialization"] = ["true"]
    default_rigid_parameter_map["AutomaticTransformInitializationMethod"] = ["GeometricalCenter"]
    # CenterOfGravity, GeometricalCenter
    
    parameter_object.AddParameterMap(default_rigid_parameter_map)
    if initial_transform is not None:
        elastix_object.SetInitialTransformParameterObject(initial_transform)

    parameter_object.SetParameter("WriteResultImage", "false")
    parameter_object.SetParameter("ComputeZYX", "true")
    parameter_object.SetParameter("MaximumStepLength", "0.01")
    parameter_object.SetParameter("MaximumNumberOfIterations", "1000")
    parameter_object.SetParameter("NumberOfResolutions", "3")
    parameter_object.SetParameter("NumberOfSpatialSamples", "10000")
    parameter_object.SetParameter("UseMultiThreadingForMetrics", "true")
    
    elastix_object.SetParameterObject(parameter_object)

    # Additional parameters
    elastix_object.SetLogToConsole(True)
    if log_file_path:
        print("log_path : ", log_file_path)
        log_directory = os.path.dirname(log_file_path)
        elastix_object.LogToFileOn()
        elastix_object.SetLogFileName(log_file_path)
        elastix_object.SetOutputDirectory(log_directory)
        print("pass log file")
    else:
        elastix_object.LogToConsoleOn()

    # Execute registration
    elastix_object.UpdateLargestPossibleRegion()

    TransParamObj = elastix_object.GetTransformParameterObject()

    return TransParamObj

def get_corresponding_file(folder, patient_id, modality):
    """Get the corresponding file for a given patient ID and modality."""
    for root, _, files in os.walk(folder):
        for file in files:
            if file.startswith(patient_id) and modality in file and file.endswith(".nii.gz"):
                return os.path.join(root, file)
    return None

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
                    print(f"Registering CBCT: {cbct_path} with MRI: {mri_path}")
                    cbct_image = itk.imread(cbct_path, itk.F)
                    mri_image = itk.imread(mri_path, itk.F)
                    print(f"CBCT size: {cbct_image.GetLargestPossibleRegion().GetSize()}")
                    print(f"MRI size: {mri_image.GetLargestPossibleRegion().GetSize()}")
                    log_file_path = os.path.join(output_folder, f'{patient_id}_registration.log')

                    TransformObj_Fine = ElastixReg(cbct_image, mri_image, log_file_path=log_file_path)

                    transforms_Fine = MatrixRetrieval(TransformObj_Fine)
                    Transforms.append(transforms_Fine)
                    transform = ComputeFinalMatrix(Transforms)
                    output_transform_path = os.path.join(output_folder, f'{patient_id}_reg.tfm')
                    print("output_transform_path : ",output_transform_path)
                    sitk.WriteTransform(transform, output_transform_path)
                    
                    print(f"Saved transform to {output_transform_path}\n")
                    
                    # Apply transform and save the transformed image
                    # intermediate_image_path = os.path.join(output_folder, f'{patient_id}_transformed.nii.gz')
                    # apply_transform_and_save(output_transform_path, mri_path, output_folder)
                    
                    # result_image_path = os.path.join(output_folder, "result.0.nii")
                    # if os.path.exists(result_image_path):
                    #     new_result_image_path = os.path.join(output_folder, f'result{patient_id}.nii')
                    #     os.rename(result_image_path, new_result_image_path)
                    #     print(f"Renamed result image to {new_result_image_path}\n")
                    # else:
                    #     print("Result image not found; skipping renaming.\n")
                    
                    # intermediate_image = itk.imread(intermediate_image_path, itk.F)
                    # log_file_path_2 = os.path.join(output_folder, f'{patient_id}_registration_2.log')

                    # TransformObj_Second = SecondElastixReg(cbct_image, intermediate_image, log_file_path=log_file_path_2)

                    # transforms_Second = MatrixRetrieval(TransformObj_Second)
                    # Transforms.append(transforms_Second)
                    # final_transform = ComputeFinalMatrix(Transforms)
                    # output_transform_path_2 = os.path.join(output_folder, f'{patient_id}_reg_2.tfm')
                    # sitk.WriteTransform(final_transform, output_transform_path_2)

                    # # Apply the final transform and save the result
                    # final_image_path = os.path.join(output_folder, f'{patient_id}_transformed_2.nii.gz')
                    # apply_transform_and_save(output_transform_path_2, intermediate_image_path, final_image_path)

                    # print(f"Saved final transformed image to {final_image_path}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Register CBCT images with corresponding MRI images.')
    parser.add_argument('--cbct_folder', type=str, help='Path to the folder containing CBCT images')
    parser.add_argument('--mri_folder', type=str, help='Path to the folder containing MRI images')
    parser.add_argument('--output_folder', type=str, help='Path to the folder where output transforms will be saved')
    
    args = parser.parse_args()
    main(args.cbct_folder, args.mri_folder, args.output_folder)