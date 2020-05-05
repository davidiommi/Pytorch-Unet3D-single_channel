
class InitParser(object):
    def __init__(self):

        self.do_you_wanna_train = False                                                # 'Training will start'
        self.do_you_wanna_load_weights = False                                        # 'Load weights'
        self.do_you_wanna_check_accuracy = True                                      # 'Model will be tested after the training or only this is done'

        # gpu setting
        self.gpu_id = 2                                                               # 'Select the GPU'
        # optimizer setting
        self.lr = 1e-4
        self.momentum = 0.9
        self.weight_decay = 1e-4
        # train setting
        self.resample = False                                                         # 'Decide or not to resample the images to a new resolution'
        self.new_resolution = (0.9375, 0.9375, 3.0)                                   # 'New resolution'
        self.patch_size = [128, 128, 64]                                                # "Input dimension for the Unet3D"
        self.drop_ratio = 0                                                           # "Probability to drop a cropped area if the label is empty. All empty patches will be dropped for 0 and accept all cropped patches if set to 1"
        self.min_pixel = 0.1                                                          # "Percentage of minimum non-zero pixels in the cropped label"
        self.batch_size = 1
        self.num_epoch = 200                                                           # "Number of training images per epoch"
        self.init_epoch = 1
        self.stride_inplane = 64                                                      # "Stride size in 2D plane"
        self.stride_layer = 32                                                        # "Stride size in z direction"

        # path setting
        self.data_path = './Data_folder/carotid/train_set/'
        self.test_path = './Data_folder/carotid/test_set/'
        self.val_path = './Data_folder/carotid/validation_set/'
        self.history_dir = './History'
        self.load_path = "./History/Network_{}.pth.gz".format(self.init_epoch-1)
        self.output_path = "./History/"

