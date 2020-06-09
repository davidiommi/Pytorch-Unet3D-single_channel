
import os
import shutil
from time import time
import re
import numpy as np
import SimpleITK as sitk
import scipy.ndimage as ndimage

def numericalSort(value):
    numbers = re.compile(r'(\d+)')
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts


def lstFiles(Path):

    images_list = []  # create an empty list, the raw image data files is stored here
    for dirName, subdirList, fileList in os.walk(Path):
        for filename in fileList:
            if ".nii.gz" in filename.lower():
                images_list.append(os.path.join(dirName, filename))
            elif ".nii" in filename.lower():
                images_list.append(os.path.join(dirName, filename))
            elif ".mhd" in filename.lower():
                images_list.append(os.path.join(dirName, filename))

    images_list = sorted(images_list, key=numericalSort)
    return images_list


list_images=lstFiles('./volumes_adjusted')
list_labels=lstFiles('./labels')

if not os.path.isdir('./Data_folder'):
    os.mkdir('./Data_folder')

for i in range(len(list_images)):
    a = list_images[i]
    b = list_labels[i]

    print(a)

    save_directory = os.path.join(str('./Data_folder/patient_' + str(i)))

    if not os.path.isdir(save_directory):
        os.mkdir(save_directory)

    label = sitk.ReadImage(b)
    image = sitk.ReadImage(a)

    label_directory = os.path.join(str(save_directory), 'label.nii')
    image_directory = os.path.join(str(save_directory), 'image.nii')

    sitk.WriteImage(image, image_directory)
    sitk.WriteImage(label, label_directory)

