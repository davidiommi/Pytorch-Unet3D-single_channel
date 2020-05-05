# 3D-Unet: patched based Pytorch implementation for medical images segmentation

3D-Unet pipeline is a computational toolbox for segmentation using neural networks. 

The training and the inference are patch based: the script randomly extract corresponding patches of the images and labels and feed them to the network during training.
The inference script extract, segment the sigle patches and automatically recontruct them in the original size.

### Example images

Sample MR images from the sagittal and coronal views for carotid artery segmentation (the segmentation result is highlighted in green)

![MR3](images/3.JPG)![MR4](images/4.JPG)
*******************************************************************************

### Requirements
pillow
scikit-learn
simpleITK
keras
scikit-image
pandas
pydicom
nibabel
tqdm
torch>=0.4.1
torchvision>=0.2.1
dominate>=2.3.1
visdom>=0.1.8.

### Python scripts and their function

- NiftiDataset.py : They augment the data, extract the patches and feed them to the GAN (reads .nii files). NiftiDataset.py
  skeleton taken from https://github.com/jackyko1991/unet3d-pytorch

- check_loader_patches: Shows example of patches fed to the network during the training  

- unet3d.py: the architecture of the U-net.

- utils.py : list of metrics and loss functions for the training

- main.py: Runs the training and the prediction on the training and validation dataset.

- predict.py: It launches the inference on training and validation data in the main.py

- predict_single_image.py: It launches the inference on a single input image chosen by the user.

## Usage
Modify the init.py to set the parameters and start the training/testing on the data:
Folder Structure:

	.
	├── Data_folder                   
	|   ├── train_set              
	|   |   ├── data_1             # Training
	|   |   |   ├── A              # Contains domain A images 
	|   |   |   └── B              # Contains domain B labels 
	|   |   └── data_2             
	|   |   |   ├── A              
	|   |   |   └── B              
	|   ├── test_set               
	|   |   ├── data_1             # Testing
	|   |   |   ├── A              
	|   |   |   └── B              
	|   |   └── data_2             
	|   |   |   ├── A              
	|   |   |   └── B              
	|   ├── validation_set               
	|   |   ├── data_1             # Validation
	|   |   |   ├── A             
	|   |   |   └── B              
	|   |   └── data_2             
	|   |   |   ├── A              
	|   |   |   └── B              

## Features
- 3D data processing ready
- Augmented patching technique, requires less image input for training
- one channel output (multichannel to be developed)
- Generic image reader with SimpleITK support (Currently only support .nii/.nii.gz format for convenience, easy to expand to DICOM, tiff and jpg format)
- Medical image pre-post processing with SimpleITK filters
- Easy network replacement structure
- Dice score similarity measurement as golden standard in medical image segmentation benchmarking

