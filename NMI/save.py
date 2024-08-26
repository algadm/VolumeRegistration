volumeNode = slicer.util.getNode('B015_MR_OR_cropLeft')
transformNode = slicer.util.getNode('B015_reg')

# Apply the transform to the volume
slicer.vtkSlicerTransformLogic().hardenTransform(volumeNode)

# Save the transformed volume
slicer.util.saveNode(volumeNode, '/home/lucia/Documents/Alban/NMI/images/MRI_mat_applied/B015_MR_OR_cropleft.nii.gz')  # Replace with your desired path and filename
