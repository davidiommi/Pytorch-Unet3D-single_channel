from NiftiDataset import *
import argparse


'''Check if the images and the labels have different size after resampling (or not) them to the same resolution'''

parser = argparse.ArgumentParser()
parser.add_argument("--images_folder", type=str, default='./Data_folder/train_set', help='path to the .nii images')
parser.add_argument("--resample", action='store_true', default=True, help='Decide or not to resample the images to a new resolution')
parser.add_argument("--new_resolution", type=float, default=(0.4, 0.4, 2.2), help='New resolution')
args = parser.parse_args()

train_list = create_list(args.images_folder)

for i in range(len(train_list)):

    a = sitk.ReadImage(train_list[i]["data"])
    if args.resample is True:
        a = resample_sitk_image(a, spacing=args.new_resolution, interpolator='linear')
    spacing1 = a.GetSpacing()
    a = sitk.GetArrayFromImage(a)
    a = np.transpose(a, axes=(2, 1, 0))  # reshape array from itk z,y,x  to  x,y,z
    a1 = a.shape

    b = sitk.ReadImage(train_list[i]["label"])
    if args.resample is True:
        b = resample_sitk_image(b, spacing=args.new_resolution, interpolator='linear')
    spacing2 = b.GetSpacing()
    b = sitk.GetArrayFromImage(b)
    b = np.transpose(b, axes=(2, 1, 0))  # reshape array from itk z,y,x  to  x,y,z
    b1 = b.shape

    print(a1)

    if a1 != b1:
        print('Mismatch of size in ', train_list[i]["data"])








































# a=sitk.ReadImage('aaaaaa.nii')
# a = sitk.GetArrayFromImage(a)
# a = np.transpose(a, axes=(2, 1, 0))  # reshape array from itk z,y,x  to  x,y,z
# result = np.rot90(a, k=-1)
# fig, ax = plt.subplots(1, 1)
# tracker = IndexTracker(ax, result)
# fig.canvas.mpl_connect('scroll_event', tracker.onscroll)
# plt.show()

# a=sitk.ReadImage(labels[36])
# a = sitk.GetArrayFromImage(a)
# a = np.transpose(a, axes=(2, 1, 0))  # reshape array from itk z,y,x  to  x,y,z
# result = np.rot90(a, k=-1)
# fig, ax = plt.subplots(1, 1)
# tracker = IndexTracker(ax, result)
# fig.canvas.mpl_connect('scroll_event', tracker.onscroll)
# plt.show()



