
class InitParser(object):
    def __init__(self):

        self.do_you_wanna_train = True                                                # 'Training will start'
        self.do_you_wanna_load_weights = False                                        # 'Load weights'
        self.do_you_wanna_check_accuracy = False                                      # 'Model will be tested after the training or only this is done'

        # gpu setting
        self.multi_gpu = True                                                         # 'Decide to use one or more GPUs'
        self.gpu_id = '2,3'                                                          # 'Select the GPUs for training and testing'
        # optimizer setting
        self.lr = 0.0002                                                              # 'Learning rate'
        self.weight_decay = 1e-4                                                      # 'Weight decay'
        # train setting
        self.increase_factor_data = 4                                                 # 'Increase the data number passed each epoch'
        self.resample = True                                                         # 'Decide or not to rescale the images to a new resolution'
        self.new_resolution = (0.6, 0.6, 2.5)                                           # 'New resolution'
        self.patch_size = [128, 128, 32]                                              # "Input dimension for the Unet3D"
        self.drop_ratio = 0                                                           # "Probability to drop a cropped area if the label is empty. All empty patches will be dropped for 0 and accept all cropped patches if set to 1"
        self.min_pixel = 0.1                                                          # "Percentage of minimum non-zero pixels in the cropped label"
        self.batch_size = 8                                                           # 'Batch size: the greater more GPUs are used and faster is the training'
        self.num_epoch = 100                                                          # "Number of epochs"
        self.init_epoch = 1
        self.stride_inplane = 64                                                      # "Stride size in 2D plane"
        self.stride_layer = 16                                                        # "Stride size in z direction"

        # path setting
        self.data_path = './Data_folder/train_set/'                           # Training data folder
        self.val_path = './Data_folder/validation_set/'                       # Validation data folder
        self.test_path = './Data_folder/test_set/'                            # Testing data folder
        self.history_dir = './History'
        self.load_path = "./History/Checkpoint/Network_{}.pth.gz".format(self.init_epoch-1)
        self.output_path = "./History/"

