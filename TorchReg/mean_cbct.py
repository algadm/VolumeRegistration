import os
import torch
import argparse
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
import torch.nn.functional as F
from sklearn.model_selection import ParameterSampler
from torchreg import AffineRegistration
from matplotlib.widgets import Slider
from metrics import NMI

# Function to load all images in the folder and normalize them
def load_cbct_images(cbct_folder):
    cbct_images = []
    for filename in os.listdir(cbct_folder):
        if (filename.endswith(".nii.gz") or filename.endswith(".nii")):  # Adjust the extension based on your files
            file_path = os.path.join(cbct_folder, filename)
            img = sitk.ReadImage(file_path)
            norm_img = normalize_image(img)
            cbct_images.append((file_path, norm_img))
    return cbct_images

# Function to normalize the image intensities to the range [0, 1]
def normalize_image(image):
    image_array = sitk.GetArrayFromImage(image).astype(np.float32)
    norm_img = (image_array - image_array.min()) / (image_array.max() - image_array.min() + 1e-8)
    norm_img = sitk.GetImageFromArray(norm_img)

    # Copy the origin, spacing, and direction from the original image
    norm_img.SetOrigin(image.GetOrigin())
    norm_img.SetSpacing(image.GetSpacing())
    norm_img.SetDirection(image.GetDirection())

    return norm_img

# Function to pad an image to the largest dimensions (z, y, x)
def pad_image_to_largest(image, largest_shape):
    # Get the numpy array from the image
    image_array = sitk.GetArrayFromImage(image)
    
    # Calculate padding amounts for each dimension
    pad_z = (largest_shape[0] - image_array.shape[0]) // 2
    pad_y = (largest_shape[1] - image_array.shape[1]) // 2
    pad_x = (largest_shape[2] - image_array.shape[2]) // 2

    # Pad the array to match the largest dimensions
    padded_array = np.pad(image_array,
                          ((pad_z, largest_shape[0] - image_array.shape[0] - pad_z),
                           (pad_y, largest_shape[1] - image_array.shape[1] - pad_y),
                           (pad_x, largest_shape[2] - image_array.shape[2] - pad_x)),
                          mode='constant', constant_values=0)

    # Convert the padded array back to a SimpleITK image
    padded_image = sitk.GetImageFromArray(padded_array)
    
    # Manually copy the origin, spacing, and direction from the original image
    padded_image.SetOrigin(image.GetOrigin())
    padded_image.SetSpacing(image.GetSpacing())
    padded_image.SetDirection(image.GetDirection())
    
    return padded_image

# Function to find the largest dimensions for each axis (z, y, x)
def find_largest_dimensions(cbct_images_with_paths):
    # Get the shape of each image (in the order: z, y, x)
    image_shapes = [sitk.GetArrayFromImage(img).shape for _, img in cbct_images_with_paths]
    
    # Find the largest size in each axis (z, y, x)
    max_z = max(shape[0] for shape in image_shapes)
    max_y = max(shape[1] for shape in image_shapes)
    max_x = max(shape[2] for shape in image_shapes)
    
    return (max_z, max_y, max_x)

# Function to pad all images to the largest dimensions
def pad_all_images_to_largest(cbct_images_with_paths):
    # Find the largest dimensions for each axis
    largest_shape = find_largest_dimensions(cbct_images_with_paths)
    
    # Pad all images to match the largest dimensions (z, y, x) and keep track of their paths
    padded_images_with_paths = [(path, pad_image_to_largest(img, largest_shape)) for path, img in cbct_images_with_paths]
    
    return padded_images_with_paths

# Function to compute the voxel-wise mean of cropped images while keeping track of paths
def compute_mean_image(cropped_images):
    # Convert all cropped images to numpy arrays and compute the mean
    try:
        mean_image = np.mean([sitk.GetArrayFromImage(img) for _, img in cropped_images], axis=0)
    except:
        mean_image = np.mean([sitk.GetArrayFromImage(img) for img in cropped_images], axis=0)

    # Convert the mean numpy array back to a SimpleITK image and copy the metadata from the first image
    mean_image_itk = sitk.GetImageFromArray(mean_image)

    if isinstance(cropped_images[0], tuple):  # Handle both cases where it's (path, img) or just img
        reference_image = cropped_images[1][1]
    else:
        reference_image = cropped_images[0]

    mean_image_itk.SetOrigin(reference_image.GetOrigin())
    mean_image_itk.SetSpacing(reference_image.GetSpacing())
    mean_image_itk.SetDirection(reference_image.GetDirection())

    return mean_image_itk

# Function to register an individual CBCT to the current mean
def registration(mean, cbct_image, cbct_path, iter, patient_id, output_folder, param_sampler):
    # Save the registered image
    image_folder = os.path.join(output_folder, f'iter_{iter}')
    if not os.path.exists(image_folder):
        os.makedirs(image_folder)
    image_path = os.path.join(image_folder, f'{patient_id}_CBCT_registered.nii.gz')
    
    best_loss = 0.0  # Initialize to track the best loss
    best_image = cbct_image
    for params in param_sampler:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"\n\033[1mIteration: {iter} -- Using {device.upper()} -- Registering CBCT: {cbct_path} with mean\033[0m")

        moving = torch.from_numpy(sitk.GetArrayFromImage(cbct_image)).float().to(device)
        static = torch.from_numpy(sitk.GetArrayFromImage(mean)).float().to(device)

        # Normalize the images
        epsilon = 1e-8    # Small value to avoid division by zero
        moving_normed = (moving - moving.min()) / (moving.max() - moving.min() + epsilon)
        static_normed = (static - static.min()) / (static.max() - static.min() + epsilon)

        # Initialize NMI loss function for rigid registration
        nmi_loss_function_rigid = NMI(intensity_range=None, nbins=64, sigma=params['sigma_rigid'], use_mask=False)

        # Initialize AffineRegistration for Rigid registration
        reg_rigid = AffineRegistration(scales=(4, 2), iterations=(100, 30), is_3d=True,
                                       learning_rate=params['learning_rate_rigid'], verbose=True,
                                       dissimilarity_function=nmi_loss_function_rigid.metric, optimizer=torch.optim.Adam,
                                       with_translation=True, with_rotation=True, with_zoom=False, with_shear=False,
                                       align_corners=True, interp_mode="trilinear", padding_mode='zeros')
        
        # Perform rigid registration
        moved_image = reg_rigid(moving_normed[None, None], static_normed[None, None])[0, 0]

        # Compute the final loss
        final_loss = -nmi_loss_function_rigid.metric(moved_image[None, None], static_normed[None, None])
        print(f"Final Loss (NMI): {final_loss}")

        image = sitk.GetImageFromArray(moved_image.cpu().numpy())

        image.SetOrigin(mean.GetOrigin())
        image.SetSpacing(mean.GetSpacing())
        image.SetDirection(mean.GetDirection())

        if final_loss > best_loss and final_loss > 1e-5:
            best_loss = final_loss
            print(f"New best parameters found with loss: {best_loss}")
            
            best_image = image
            sitk.WriteImage(image, image_path)

    return best_image

# Function to display slices of the volume interactively using a slider
def display_volume_with_slider(volume_image_itk):
    # Convert the SimpleITK image to a NumPy array
    volume_array = sitk.GetArrayFromImage(volume_image_itk)
    
    # Initial slice index (middle sagittal slice)
    initial_slice = volume_array.shape[2] // 2  # For sagittal, use shape[2]

    # Create the figure and axis
    fig, ax = plt.subplots(1, 1)
    plt.subplots_adjust(bottom=0.25)  # Adjust bottom to make space for slider

    # Display the initial slice (sagittal)
    img_display = ax.imshow(volume_array[:, :, initial_slice], cmap='gray')
    ax.set_title(f'Sagittal Slice {initial_slice}')
    ax.axis('off')  # Turn off axis labels

    # Create the slider for the sagittal plane (adjust for x-axis, which is shape[2])
    ax_slider = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor='lightgoldenrodyellow')  # Position the slider
    slice_slider = Slider(ax_slider, 'Sagittal Slice', 0, volume_array.shape[2] - 1, valinit=initial_slice, valstep=1)

    # Update function for slider
    def update_slice(val):
        slice_idx = int(slice_slider.val)  # Get the current slice index
        img_display.set_data(volume_array[:, :, slice_idx])  # Update the image data for sagittal plane
        ax.set_title(f'Sagittal Slice {slice_idx}')  # Update the title with current slice number
        fig.canvas.draw_idle()  # Redraw the canvas

    # Call the update function when the slider value changes
    slice_slider.on_changed(update_slice)

    # Display the plot with the slider
    plt.show()

# Main function to iterate and compute the final average
def main(cbct_folder, output_folder, num_iterations, param_sampler):
    # Generate output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Load all CBCT images from the folder
    cbct_images = load_cbct_images(cbct_folder)

    # Pad all images to the size of the largest volume
    padded_images = pad_all_images_to_largest(cbct_images)
    
    # Compute the mean of the padded images
    mean_image = padded_images[1][1]
    mean_path = os.path.join(output_folder, 'mean_image.nii.gz')
    sitk.WriteImage(mean_image, mean_path)  # Save the initial mean
    
    for iteration in range(num_iterations):
        # Register all images to the current mean
        registered_images = []
        for path, img in padded_images:
            patient_id = os.path.basename(path).split("_CBCT_")[0]
            registered_image = registration(mean_image, img, path, iteration, patient_id, output_folder, param_sampler)
            registered_images.append((path, registered_image))

        # Update padded_images with the newly registered images
        padded_images = registered_images
        
        # Compute the new mean from the registered images
        mean_image = compute_mean_image(registered_images)
    
        # Save the final mean image
        sitk.WriteImage(mean_image, mean_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Creating CBCT mean for TMJ identification.')
    parser.add_argument('--cbct_folder', type=str, required=True, help='Path to the folder containing CBCT images')
    parser.add_argument('--output_folder', type=str, required=True, help='Path to the folder where the mean will be saved')
    parser.add_argument('--num_iterations', type=int, default=5, help='Number of iterations for iterative mean computation')
    
    args = parser.parse_args()

    # Define the parameter grid for hyperparameter search
    param_grid = {
        'learning_rate_rigid': np.logspace(-5, -3, 10),   # Learning rate for rigid registration
        'sigma_rigid': np.logspace(-2, -1, 3),            # Number of iterations for rigid registration
    }

    # Number of parameter combinations to sample
    n_samples = 30
    param_sampler = ParameterSampler(param_grid, n_iter=n_samples)

    main(args.cbct_folder, args.output_folder, args.num_iterations, param_sampler)




# import os
# import torch
# import argparse
# import numpy as np
# import SimpleITK as sitk
# import matplotlib.pyplot as plt
# import torch.nn.functional as F
# from sklearn.model_selection import ParameterSampler
# from torchreg import AffineRegistration
# from matplotlib.widgets import Slider
# from metrics import NMI

# # Function to load all images in the folder and normalize them
# def load_cbct_images(cbct_folder):
#     cbct_images = []
#     for filename in os.listdir(cbct_folder):
#         if (filename.endswith(".nii.gz") or filename.endswith(".nii")):  # Adjust the extension based on your files
#             file_path = os.path.join(cbct_folder, filename)
#             img = sitk.ReadImage(file_path)
#             norm_img = normalize_image(img)
#             cbct_images.append((file_path, norm_img))
#     return cbct_images

# # Function to normalize the image intensities to the range [0, 1]
# def normalize_image(image):
#     image_array = sitk.GetArrayFromImage(image).astype(np.float32)
#     norm_img = (image_array - image_array.min()) / (image_array.max() - image_array.min() + 1e-8)
#     norm_img = sitk.GetImageFromArray(norm_img)

#     # Copy the origin, spacing, and direction from the original image
#     norm_img.SetOrigin(image.GetOrigin())
#     norm_img.SetSpacing(image.GetSpacing())
#     norm_img.SetDirection(image.GetDirection())

#     return norm_img

# # Function to pad an image to the largest dimensions (z, y, x)
# def pad_image_to_largest(image, largest_shape):
#     # Get the numpy array from the image
#     image_array = sitk.GetArrayFromImage(image)
    
#     # Calculate padding amounts for each dimension
#     pad_z = (largest_shape[0] - image_array.shape[0]) // 2
#     pad_y = (largest_shape[1] - image_array.shape[1]) // 2
#     pad_x = (largest_shape[2] - image_array.shape[2]) // 2

#     # Pad the array to match the largest dimensions
#     padded_array = np.pad(image_array,
#                           ((pad_z, largest_shape[0] - image_array.shape[0] - pad_z),
#                            (pad_y, largest_shape[1] - image_array.shape[1] - pad_y),
#                            (pad_x, largest_shape[2] - image_array.shape[2] - pad_x)),
#                           mode='constant', constant_values=0)

#     # Convert the padded array back to a SimpleITK image
#     padded_image = sitk.GetImageFromArray(padded_array)
    
#     # Manually copy the origin, spacing, and direction from the original image
#     padded_image.SetOrigin(image.GetOrigin())
#     padded_image.SetSpacing(image.GetSpacing())
#     padded_image.SetDirection(image.GetDirection())
    
#     return padded_image

# # Function to find the largest dimensions for each axis (z, y, x)
# def find_largest_dimensions(cbct_images_with_paths):
#     # Get the shape of each image (in the order: z, y, x)
#     image_shapes = [sitk.GetArrayFromImage(img).shape for _, img in cbct_images_with_paths]
    
#     # Find the largest size in each axis (z, y, x)
#     max_z = max(shape[0] for shape in image_shapes)
#     max_y = max(shape[1] for shape in image_shapes)
#     max_x = max(shape[2] for shape in image_shapes)
    
#     return (max_z, max_y, max_x)

# # Function to pad all images to the largest dimensions
# def pad_all_images_to_largest(cbct_images_with_paths):
#     # Find the largest dimensions for each axis
#     largest_shape = find_largest_dimensions(cbct_images_with_paths)
    
#     # Pad all images to match the largest dimensions (z, y, x) and keep track of their paths
#     padded_images_with_paths = [(path, pad_image_to_largest(img, largest_shape)) for path, img in cbct_images_with_paths]
    
#     return padded_images_with_paths

# # Function to compute the voxel-wise mean of cropped images while keeping track of paths
# def compute_mean_image(cropped_images):
#     # Convert all cropped images to numpy arrays and compute the mean
#     try:
#         mean_image = np.mean([sitk.GetArrayFromImage(img) for _, img in cropped_images], axis=0)
#     except:
#         mean_image = np.mean([sitk.GetArrayFromImage(img) for img in cropped_images], axis=0)

#     # Convert the mean numpy array back to a SimpleITK image and copy the metadata from the first image
#     mean_image_itk = sitk.GetImageFromArray(mean_image)

#     if isinstance(cropped_images[0], tuple):  # Handle both cases where it's (path, img) or just img
#         reference_image = cropped_images[1][1]
#     else:
#         reference_image = cropped_images[0]

#     mean_image_itk.SetOrigin(reference_image.GetOrigin())
#     mean_image_itk.SetSpacing(reference_image.GetSpacing())
#     mean_image_itk.SetDirection(reference_image.GetDirection())

#     return mean_image_itk

# # Function to register an individual CBCT to the current mean
# def registration(mean, cbct_image, cbct_path, iter, patient_id, output_folder):
#     # Save the registered image
#     image_folder = os.path.join(output_folder, f'iter_{iter}')
#     if not os.path.exists(image_folder):
#         os.makedirs(image_folder)
#     image_path = os.path.join(image_folder, f'{patient_id}_CBCT_registered.nii.gz')
    
#     device = 'cuda' if torch.cuda.is_available() else 'cpu'
#     print(f"\n\033[1mIteration: {iter} -- Using {device.upper()} -- Registering CBCT: {cbct_path} with mean\033[0m")

#     moving = torch.from_numpy(sitk.GetArrayFromImage(cbct_image)).float().to(device)
#     static = torch.from_numpy(sitk.GetArrayFromImage(mean)).float().to(device)

#     # Normalize the images
#     epsilon = 1e-8    # Small value to avoid division by zero
#     moving_normed = (moving - moving.min()) / (moving.max() - moving.min() + epsilon)
#     static_normed = (static - static.min()) / (static.max() - static.min() + epsilon)

#     # Initialize NMI loss function for rigid registration
#     nmi_loss_function_rigid = NMI(intensity_range=None, nbins=64, sigma=0.1, use_mask=False)

#     # Initialize AffineRegistration for Rigid registration
#     reg_rigid = AffineRegistration(scales=(4, 2), iterations=(1000, 300), is_3d=True, learning_rate=1e-5, verbose=True,
#                                    dissimilarity_function=nmi_loss_function_rigid.metric, optimizer=torch.optim.Adam,
#                                    with_translation=True, with_rotation=True, with_zoom=True, with_shear=False,
#                                    align_corners=True, interp_mode="trilinear", padding_mode='zeros')
    
#     # Perform rigid registration
#     moved_image = reg_rigid(moving_normed[None, None], static_normed[None, None])[0, 0]

#     # Compute the final loss
#     final_loss = -nmi_loss_function_rigid.metric(moved_image[None, None], static_normed[None, None])
#     print(f"Final Loss (NMI): {final_loss}")

#     image = sitk.GetImageFromArray(moved_image.cpu().numpy())

#     image.SetOrigin(mean.GetOrigin())
#     image.SetSpacing(mean.GetSpacing())
#     image.SetDirection(mean.GetDirection())

#     sitk.WriteImage(image, image_path)

#     return image

# # Function to display slices of the volume interactively using a slider
# def display_volume_with_slider(volume_image_itk):
#     # Convert the SimpleITK image to a NumPy array
#     volume_array = sitk.GetArrayFromImage(volume_image_itk)
    
#     # Initial slice index (middle sagittal slice)
#     initial_slice = volume_array.shape[2] // 2  # For sagittal, use shape[2]

#     # Create the figure and axis
#     fig, ax = plt.subplots(1, 1)
#     plt.subplots_adjust(bottom=0.25)  # Adjust bottom to make space for slider

#     # Display the initial slice (sagittal)
#     img_display = ax.imshow(volume_array[:, :, initial_slice], cmap='gray')
#     ax.set_title(f'Sagittal Slice {initial_slice}')
#     ax.axis('off')  # Turn off axis labels

#     # Create the slider for the sagittal plane (adjust for x-axis, which is shape[2])
#     ax_slider = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor='lightgoldenrodyellow')  # Position the slider
#     slice_slider = Slider(ax_slider, 'Sagittal Slice', 0, volume_array.shape[2] - 1, valinit=initial_slice, valstep=1)

#     # Update function for slider
#     def update_slice(val):
#         slice_idx = int(slice_slider.val)  # Get the current slice index
#         img_display.set_data(volume_array[:, :, slice_idx])  # Update the image data for sagittal plane
#         ax.set_title(f'Sagittal Slice {slice_idx}')  # Update the title with current slice number
#         fig.canvas.draw_idle()  # Redraw the canvas

#     # Call the update function when the slider value changes
#     slice_slider.on_changed(update_slice)

#     # Display the plot with the slider
#     plt.show()

# # Main function to iterate and compute the final average
# def main(cbct_folder, output_folder, num_iterations):
#     # Generate output folder if it doesn't exist
#     if not os.path.exists(output_folder):
#         os.makedirs(output_folder)

#     # Load all CBCT images from the folder
#     cbct_images = load_cbct_images(cbct_folder)

#     # Pad all images to the size of the largest volume
#     padded_images = pad_all_images_to_largest(cbct_images)
    
#     # Compute the mean of the padded images
#     mean_image = padded_images[1][1]
#     mean_path = os.path.join(output_folder, 'mean_image.nii.gz')
#     sitk.WriteImage(mean_image, mean_path)  # Save the initial mean
    
#     for iteration in range(num_iterations):
#         # Register all images to the current mean
#         registered_images = []
#         for path, img in padded_images:
#             patient_id = os.path.basename(path).split("_CBCT_")[0]
#             registered_image = registration(mean_image, img, path, iteration, patient_id, output_folder)
#             registered_images.append((path, registered_image))

#         # Update padded_images with the newly registered images
#         padded_images = registered_images
        
#         # Compute the new mean from the registered images
#         mean_image = compute_mean_image(registered_images)
    
#         # Save the final mean image
#         sitk.WriteImage(mean_image, mean_path)

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description='Creating CBCT mean for TMJ identification.')
#     parser.add_argument('--cbct_folder', type=str, required=True, help='Path to the folder containing CBCT images')
#     parser.add_argument('--output_folder', type=str, required=True, help='Path to the folder where the mean will be saved')
#     parser.add_argument('--num_iterations', type=int, default=5, help='Number of iterations for iterative mean computation')
    
#     args = parser.parse_args()

#     main(args.cbct_folder, args.output_folder, args.num_iterations)