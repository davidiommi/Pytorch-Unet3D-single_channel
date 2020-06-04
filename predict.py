from NiftiDataset import *
import NiftiDataset as NiftiDataset
from tqdm import tqdm
import datetime
from predict_single_image import from_numpy_to_itk, prepare_batch, inference
import math


def inference_all(model, image_list, resample, resolution, patch_size_x, patch_size_y, patch_size_z, stride_inplane, stride_layer, batch_size, segmentation):

    image = (image_list["data"])
    label = (image_list["label"])

    # a = (image.split('/')[-1]) # my pc
    # a = (a.split('\\')[-2])

    a = (image.split('/')[-2])  # dgx

    if not os.path.isdir('./Data_folder/results'):
        os.mkdir('./Data_folder/results')

    label_directory = os.path.join(str('./Data_folder/results/results_' + a + '.nii'))

    result, dice = inference(False, model, image, label, './prova.nii', resample, resolution, patch_size_x,
                       patch_size_y, patch_size_z, stride_inplane, stride_layer, batch_size, segmentation=segmentation)

    # save segmented label
    writer = sitk.ImageFileWriter()
    writer.SetFileName(label_directory)
    writer.Execute(result)
    print("{}: Save evaluate label at {} success".format(datetime.datetime.now(), label_directory))
    print('************* Next image coming... *************')

    return dice


def check_accuracy_model(model, images, resample, new_resolution, patch_size_x, patch_size_y, patch_size_z, stride_inplane, stride_layer):

    np_dice = []
    print("0/%i (0%%)" % len(images))
    for i in range(len(images)):

       Np_dice = inference_all(model=model, image_list=images[i], resample=resample, resolution=new_resolution, patch_size_x=patch_size_x,
                                        patch_size_y=patch_size_y, patch_size_z=patch_size_z,  stride_inplane=stride_inplane, stride_layer=stride_layer, batch_size=1, segmentation=True)

       np_dice.append(Np_dice)

    np_dice = np.array(np_dice)
    print('Mean volumetric DSC:', np_dice.mean())
    return np_dice.mean()
