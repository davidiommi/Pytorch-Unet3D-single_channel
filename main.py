import time
import os
import torch
from predict import *
from torch.utils.data import DataLoader
from torch.autograd import Variable
from utils import AvgMeter, dice_coeff, check_dir, DiceLoss, BinaryDiceLoss
from init import InitParser
from NiftiDataset import *
import NiftiDataset as NiftiDataset
from UNet import UNet


def test_epoch(net, loader):
    # we transfer the mode of network to test
    net.eval()
    test_dice_meter = AvgMeter()
    for batch_idx, (data, label) in enumerate(loader):
        data = Variable(data.cuda())
        output = net(data)

        output = output.squeeze().data.cpu().numpy()
        label = label.squeeze().cpu().numpy()

        test_dice_meter.update(dice_coeff(output, label))

        # print("Test {} || Dice: {:.4f}".format(str(batch_idx).zfill(4), test_dice_meter.val))
    return test_dice_meter.avg


def train_epoch(net, loader, optimizer, cost):
    # we transfer the mode of network to train
    net.train()

    batch_loss = AvgMeter()
    for batch_idx, (data, label) in enumerate(loader):
        data = Variable(data.cuda())                                                       # A Variable wraps a Tensor. It supports nearly all the API’s defined by a Tensor.
        label = Variable(label.cuda())

        output = net(data)                                                                 # Give the data to the network

        loss = cost(output, label)
        # evaluate the cost function
        output = output.squeeze().data.cpu().numpy()
        label = label.squeeze().cpu().numpy()
        dice = dice_coeff(output, label)

        optimizer.zero_grad()                                                              # we need to set the gradients to zero before starting to do backpropragation because PyTorch accumulates the gradients on subsequent backward passes
        loss.backward()
        optimizer.step()

        batch_loss.update(loss.item())
        if batch_idx % 10 == 0:
            print("Train Batch {} || Loss: {:.4f} | Training Dice: {:.4f}".format(str(batch_idx).zfill(4), batch_loss.val, dice))
    return batch_loss.avg


def main(args):
    ckpt_path = os.path.join(args.output_path, "Checkpoint")
    log_path = os.path.join(args.output_path, "Log")

    min_pixel = int(args.min_pixel * ((args.patch_size[0] * args.patch_size[1] * args.patch_size[2]) / 100))

    check_dir(args.output_path)
    check_dir(log_path)
    check_dir(ckpt_path)

    if args.do_you_wanna_train is True:

        train_list = create_list(args.data_path)
        val_list = create_list(args.val_path)
        test_list = create_list(args.test_path)

        for i in range(args.increase_factor_data):                                                # augment the data list for training

            train_list.extend(train_list)
            val_list.extend(val_list)
            test_list.extend(test_list)

        print('Number of training patches per epoch:', len(train_list))
        print('Number of validation patches per epoch:', len(val_list))
        print('Number of testing patches per epoch:', len(test_list))

        trainTransforms = [
            NiftiDataset.Resample(args.new_resolution, args.resample),
            NiftiDataset.Augmentation(),
            NiftiDataset.Padding((args.patch_size[0], args.patch_size[1], args.patch_size[2])),
            NiftiDataset.RandomCrop((args.patch_size[0], args.patch_size[1], args.patch_size[2]), args.drop_ratio,
                                    min_pixel),
        ]

        valTransforms = [
            NiftiDataset.Resample(args.new_resolution, args.resample),
            NiftiDataset.Padding((args.patch_size[0], args.patch_size[1], args.patch_size[2])),
            NiftiDataset.RandomCrop((args.patch_size[0], args.patch_size[1], args.patch_size[2]), args.drop_ratio,
                                    min_pixel),
        ]

        # define the dataset and loader
        train_set = NifitDataSet(train_list, transforms=trainTransforms, train=True)
        val_set = NifitDataSet(val_list, transforms=valTransforms, test=True)
        test_set = NifitDataSet(test_list, transforms=valTransforms, test=True)

        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)                                   # Here are then fed to the network with a defined batch size
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)
        test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

        # define the network and load the init weight
        if args.multi_gpu is True:
            os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id  # Multi-gpu selector for training
            net = torch.nn.DataParallel((UNet(residual='pool')).cuda())  # load the network Unet

        else:
            torch.cuda.set_device(args.gpu_id)
            net = UNet(residual='pool').cuda()

        if args.do_you_wanna_load_weights is True:
            net.load_state_dict(torch.load(args.load_path))                                    # load the weights of the network if you have it

        # define the optimizer of the training process                                         # define the optimizer of the training process
        optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        # define the loss function
        cost = BinaryDiceLoss()                                                              # define the loss function
        best_dice = 0.
        for epoch in range(args.init_epoch, args.init_epoch+args.num_epoch):                   # define the epochs number
            start_time = time.time()
            # train one epoch
            epoch_loss = train_epoch(net, train_loader, optimizer, cost)                       # training function
            # eval in test data after one epoch training

            # epoch_dice_training = test_epoch(net, train_loader)
            epoch_dice_val = test_epoch(net, val_loader)
            epoch_dice_test = test_epoch(net, test_loader)

            epoch_time = time.time() - start_time

            info_line = "Epoch {} || Loss: {:.4f} | Time(min): {:.2f} |Validation Dice: {:.4f} | Testing Dice: {:.4f}".format(
                str(epoch).zfill(3), epoch_loss, epoch_time/60, epoch_dice_val, epoch_dice_test
            )
            print(info_line)
            open(os.path.join(log_path, 'train_log.txt'), 'a').write(info_line+'\n')

            # save the checkpoint
            torch.save(net.state_dict(), os.path.join(ckpt_path, "Network_{}.pth.gz".format(epoch)))
            if epoch_dice_val > best_dice:
                best_dice = epoch_dice_val
                torch.save(net.state_dict(), os.path.join(ckpt_path, "Best_Dice.pth.gz"))

    if args.do_you_wanna_check_accuracy is True:

        if args.multi_gpu is True:
            os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id  # Multi-gpu selector for training
            net = torch.nn.DataParallel((UNet(residual='pool')).cuda())  # load the network Unet

        else:
            torch.cuda.set_device(args.gpu_id)
            net = UNet(residual='pool').cuda()

        net.load_state_dict(torch.load('./History/Checkpoint/Best_Dice.pth.gz'))

        train_list = create_list(args.data_path)
        val_list = create_list(args.val_path)
        test_list = create_list(args.test_path)

        print("Checking accuracy on validation set")
        Dice_val = check_accuracy_model(net, val_list, args.resample, args.new_resolution, args.patch_size[0],
                                        args.patch_size[1], args.patch_size[2],
                                        args.stride_inplane, args.stride_layer)

        print("Checking accuracy on testing set")
        Dice_test = check_accuracy_model(net, test_list, args.resample, args.new_resolution, args.patch_size[0],
                                        args.patch_size[1], args.patch_size[2],
                                        args.stride_inplane, args.stride_layer)

        print("Checking accuracy on training set")
        Dice_train = check_accuracy_model(net, train_list, args.resample, args.new_resolution, args.patch_size[0],
                                        args.patch_size[1], args.patch_size[2],
                                        args.stride_inplane, args.stride_layer)

        print("Dice_val:",Dice_val,"Dice_test:",Dice_test,"Dice_train:",Dice_train)


if __name__ == '__main__':
    parsers = InitParser()
    main(parsers)
