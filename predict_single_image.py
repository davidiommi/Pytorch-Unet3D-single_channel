#!/usr/bin/env python2
# -*- coding: utf-8 -*-
from NiftiDataset import *
import NiftiDataset as NiftiDataset
from tqdm import tqdm
import datetime
import argparse
import matplotlib.pyplot as plt
import math
import scipy
from UNet import UNet
import torch
from utils import *
from torch.autograd import Variable

parser = argparse.ArgumentParser()
parser.add_argument('--multi_gpu', default=False, help='Multi or single Gpu')
parser.add_argument('--gpu_id', default='0', help='Select the GPU')
parser.add_argument("--image", type=str, default='./Data_folder/carotid/train_set/data_1/image.nii')
parser.add_argument("--label", type=str, default='./Data_folder/carotid/train_set/data_1/label.nii')
parser.add_argument("--result", type=str, default='./prova.nii', help='path to the .nii result to save')
parser.add_argument("--weights", type=str, default='./History/Checkpoint/Best_Dice.pth.gz', help='generator weights to load')
parser.add_argument("--resample", default=False, help='Decide or not to resample the images to a new resolution')
parser.add_argument("--new_resolution", type=float, default=(1, 1, 2), help='New resolution')
parser.add_argument("--patch_size", type=int, nargs=3, default=[64, 64, 64], help="Input dimension for the generator")
parser.add_argument("--batch_size", type=int, nargs=1, default=1, help="Batch size to feed the network (currently supports 1)")
parser.add_argument("--stride_inplane", type=int, nargs=1, default=32, help="Stride size in 2D plane")
parser.add_argument("--stride_layer", type=int, nargs=1, default=32, help="Stride size in z direction")
args = parser.parse_args()

def from_numpy_to_itk(image_np,image_itk):
    image_np = np.transpose(image_np, (2, 1, 0))
    image = sitk.GetImageFromArray(image_np)
    image.SetOrigin(image_itk.GetOrigin())
    image.SetDirection(image_itk.GetDirection())
    image.SetSpacing(image_itk.GetSpacing())
    return image


def prepare_batch(image, ijk_patch_indices):
    image_batches = []
    for batch in ijk_patch_indices:
        image_batch = []
        for patch in batch:
            image_patch = image[patch[0]:patch[1], patch[2]:patch[3], patch[4]:patch[5]]
            image_batch.append(image_patch)

        image_batch = np.asarray(image_batch)
        # image_batch = image_batch[:, :, :, :, np.newaxis]
        image_batches.append(image_batch)

    return image_batches


# inference single image
def inference(write_image, model, image_path, label_path, result_path, resample, resolution, patch_size_x, patch_size_y, patch_size_z, stride_inplane, stride_layer, batch_size=1, segmentation=True):

    # create transformations to image and labels
    transforms1 = [
        NiftiDataset.Resample(resolution, resample)
    ]

    transforms2 = [
        NiftiDataset.Padding((patch_size_x, patch_size_y, patch_size_z))
    ]

    # read image file
    reader = sitk.ImageFileReader()
    reader.SetFileName(image_path)
    image = reader.Execute()

    # normalize the image
    image = Normalization(image)

    castImageFilter = sitk.CastImageFilter()
    castImageFilter.SetOutputPixelType(sitk.sitkFloat32)
    image = castImageFilter.Execute(image)

    # create empty label in pair with transformed image
    label_tfm = sitk.Image(image.GetSize(), sitk.sitkFloat32)
    label_tfm.SetOrigin(image.GetOrigin())
    label_tfm.SetDirection(image.GetDirection())
    label_tfm.SetSpacing(image.GetSpacing())

    sample = {'image': image, 'label': label_tfm}

    for transform in transforms1:
        sample = transform(sample)

    # keeping track on how much padding will be performed before the inference
    image_array = sitk.GetArrayFromImage(sample['image'])
    pad_x = patch_size_x - (patch_size_x - image_array.shape[2])
    pad_y = patch_size_x - (patch_size_y - image_array.shape[1])
    pad_z = patch_size_z - (patch_size_z - image_array.shape[0])

    image_pre_pad = sample['image']

    for transform in transforms2:
        sample = transform(sample)

    image_tfm, label_tfm = sample['image'], sample['label']

    # convert image to numpy array
    image_np = sitk.GetArrayFromImage(image_tfm)
    label_np = sitk.GetArrayFromImage(label_tfm)

    label_np = np.asarray(label_np, np.float32)

    # unify numpy and sitk orientation
    image_np = np.transpose(image_np, (2, 1, 0))
    label_np = np.transpose(label_np, (2, 1, 0))

    if segmentation is True:
        label_np = np.around(label_np)

    # ----------------- Padding the image if the z dimension still is not even ----------------------

    if (image_np.shape[2] % 2) == 0:
        Padding = False
    else:
        image_np = np.pad(image_np, ((0,0), (0,0), (0, 1)), 'edge')
        label_np = np.pad(label_np, ((0, 0), (0, 0), (0, 1)), 'edge')
        Padding = True

    # ------------------------------------------------------------------------------------------------

    # a weighting matrix will be used for averaging the overlapped region
    weight_np = np.zeros(label_np.shape)

    # prepare image batch indices
    inum = int(math.ceil((image_np.shape[0] - patch_size_x) / float(stride_inplane))) + 1
    jnum = int(math.ceil((image_np.shape[1] - patch_size_y) / float(stride_inplane))) + 1
    knum = int(math.ceil((image_np.shape[2] - patch_size_z) / float(stride_layer))) + 1

    patch_total = 0
    ijk_patch_indices = []
    ijk_patch_indicies_tmp = []

    for i in range(inum):
        for j in range(jnum):
            for k in range(knum):
                if patch_total % batch_size == 0:
                    ijk_patch_indicies_tmp = []

                istart = i * stride_inplane
                if istart + patch_size_x > image_np.shape[0]:  # for last patch
                    istart = image_np.shape[0] - patch_size_x
                iend = istart + patch_size_x

                jstart = j * stride_inplane
                if jstart + patch_size_y > image_np.shape[1]:  # for last patch
                    jstart = image_np.shape[1] - patch_size_y
                jend = jstart + patch_size_y

                kstart = k * stride_layer
                if kstart + patch_size_z > image_np.shape[2]:  # for last patch
                    kstart = image_np.shape[2] - patch_size_z
                kend = kstart + patch_size_z

                ijk_patch_indicies_tmp.append([istart, iend, jstart, jend, kstart, kend])

                if patch_total % batch_size == 0:
                    ijk_patch_indices.append(ijk_patch_indicies_tmp)

                patch_total += 1

    batches = prepare_batch(image_np, ijk_patch_indices)

    for i in tqdm(range(len(batches))):
        batch = batches[i]

        batch = torch.from_numpy(batch[np.newaxis, :, :, :])
        batch = Variable(batch.cuda())

        pred = model(batch)
        pred = pred.squeeze().data.cpu().numpy()

        istart = ijk_patch_indices[i][0][0]
        iend = ijk_patch_indices[i][0][1]
        jstart = ijk_patch_indices[i][0][2]
        jend = ijk_patch_indices[i][0][3]
        kstart = ijk_patch_indices[i][0][4]
        kend = ijk_patch_indices[i][0][5]
        label_np[istart:iend, jstart:jend, kstart:kend] += pred[:, :, :]
        weight_np[istart:iend, jstart:jend, kstart:kend] += 1.0

    print("{}: Evaluation complete".format(datetime.datetime.now()))
    # eliminate overlapping region using the weighted value
    label_np = (np.float32(label_np) / np.float32(weight_np) + 0.01)

    if segmentation is True:
        label_np = abs(np.around(label_np))

    # removed the 1 pad on z
    if Padding is True:
        label_np = label_np[:, :, 0:(label_np.shape[2]-1)]

    # removed all the padding
    label_np = label_np[:pad_x, :pad_y, :pad_z]

    # convert back to sitk space
    label = from_numpy_to_itk(label_np, image_pre_pad)
    # ---------------------------------------------------------------------------------------------

    # save label
    writer = sitk.ImageFileWriter()

    if resample is True:

        print("{}: Resampling label back to original image space...".format(datetime.datetime.now()))
        # label = resample_sitk_image(label, spacing=image.GetSpacing(), interpolator='bspline')   # keep this commented
        if segmentation is True:
            label = resize(label, (sitk.GetArrayFromImage(image)).shape[::-1], sitk.sitkBSpline)
            label_array = np.around(sitk.GetArrayFromImage(label))
            label = sitk.GetImageFromArray(label_array)
            label.SetDirection(image.GetDirection())
            label.SetOrigin(image.GetOrigin())
            label.SetSpacing(image.GetSpacing())
        else:
            label = resize(label, (sitk.GetArrayFromImage(image)).shape[::-1], sitk.sitkBSpline)
            label.SetDirection(image.GetDirection())
            label.SetOrigin(image.GetOrigin())
            label.SetSpacing(image.GetSpacing())


    else:
        label = label

    if label_path is not None and segmentation is True:

        reader = sitk.ImageFileReader()
        reader.SetFileName(label_path)
        true_label = reader.Execute()

        true_label = sitk.GetArrayFromImage(true_label)
        predicted = sitk.GetArrayFromImage(label)

        dice = dice_coeff(predicted,true_label)


    writer.SetFileName(result_path)
    if write_image is True:
        writer.Execute(label)
        print("{}: Save evaluate label at {} success".format(datetime.datetime.now(), result_path))

    if label_path is not None and segmentation is True:
        print("Dice score:", dice)

        return label, dice

    else:
        dice = None
        return label, dice


if __name__ == "__main__":

    if args.multi_gpu is True:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id  # Multi-gpu selector for training
        net = torch.nn.DataParallel((UNet(residual='pool')).cuda())  # load the network Unet

    else:
        torch.cuda.set_device(args.gpu_id)
        net = UNet(residual='pool').cuda()

    net.load_state_dict(torch.load(args.weights))

    result, dice = inference(True, net, args.image, None, args.result, args.resample, args.new_resolution,
                       args.patch_size[0],args.patch_size[1],args.patch_size[2], args.stride_inplane, args.stride_layer, segmentation=True)
