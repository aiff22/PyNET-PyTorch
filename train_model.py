# Copyright 2020 by Andrey Ignatov. All Rights Reserved.

from torch.utils.data import DataLoader
from torchvision import transforms
from torch.optim import Adam

import torch
import imageio
import numpy as np
import math
import sys

from load_data import LoadData, LoadVisualData
from msssim import MSSSIM
from model import PyNET
from vgg import vgg_19
from utils import normalize_batch, process_command_args

to_image = transforms.Compose([transforms.ToPILImage()])

np.random.seed(0)
torch.manual_seed(0)

# Processing command arguments

level, batch_size, learning_rate, restore_epoch, num_train_epochs, dataset_dir = process_command_args(sys.argv)
dslr_scale = float(1) / (2 ** (level - 1))

# Dataset size

TRAIN_SIZE = 46839
TEST_SIZE = 1204


def train_model():

    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda")

    print("CUDA visible devices: " + str(torch.cuda.device_count()))
    print("CUDA Device Name: " + str(torch.cuda.get_device_name(device)))

    # Creating dataset loaders

    train_dataset = LoadData(dataset_dir, TRAIN_SIZE, dslr_scale, test=False)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=1,
                              pin_memory=True, drop_last=True)

    test_dataset = LoadData(dataset_dir, TEST_SIZE, dslr_scale, test=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=1,
                             pin_memory=True, drop_last=False)

    visual_dataset = LoadVisualData(dataset_dir, 10, dslr_scale, level)
    visual_loader = DataLoader(dataset=visual_dataset, batch_size=1, shuffle=False, num_workers=0,
                               pin_memory=True, drop_last=False)

    # Creating image processing network and optimizer

    generator = PyNET(level=level, instance_norm=True, instance_norm_level_1=True).to(device)
    generator = torch.nn.DataParallel(generator)

    optimizer = Adam(params=generator.parameters(), lr=learning_rate)

    # Restoring the variables

    if level < 5:
        generator.load_state_dict(torch.load("models/pynet_level_" + str(level + 1) +
                                             "_epoch_" + str(restore_epoch) + ".pth"), strict=False)

    # Losses

    VGG_19 = vgg_19(device)
    MSE_loss = torch.nn.MSELoss()
    MS_SSIM = MSSSIM()

    # Train the network

    for epoch in range(num_train_epochs):

        torch.cuda.empty_cache()

        train_iter = iter(train_loader)
        for i in range(len(train_loader)):

            optimizer.zero_grad()
            x, y = next(train_iter)

            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            enhanced = generator(x)

            # MSE Loss
            loss_mse = MSE_loss(enhanced, y)

            # VGG Loss

            if level < 5:
                enhanced_vgg = VGG_19(normalize_batch(enhanced))
                target_vgg = VGG_19(normalize_batch(y))
                loss_content = MSE_loss(enhanced_vgg, target_vgg)

            # Total Loss

            if level == 5 or level == 4:
                total_loss = loss_mse
            if level == 3 or level == 2:
                total_loss = loss_mse * 10 + loss_content
            if level == 1:
                total_loss = loss_mse * 10 + loss_content
            if level == 0:
                loss_ssim = MS_SSIM(enhanced, y)
                total_loss = loss_mse + loss_content + (1 - loss_ssim) * 0.4

            # Perform the optimization step

            total_loss.backward()
            optimizer.step()

            if i == 0:

                # Save the model that corresponds to the current epoch

                generator.eval().cpu()
                torch.save(generator.state_dict(), "models/pynet_level_" + str(level) + "_epoch_" + str(epoch) + ".pth")
                generator.to(device).train()

                # Save visual results for several test images

                generator.eval()
                with torch.no_grad():

                    visual_iter = iter(visual_loader)
                    for j in range(len(visual_loader)):

                        torch.cuda.empty_cache()

                        raw_image = next(visual_iter)
                        raw_image = raw_image.to(device, non_blocking=True)

                        enhanced = generator(raw_image.detach())
                        enhanced = np.asarray(to_image(torch.squeeze(enhanced.detach().cpu())))

                        imageio.imwrite("results/pynet_img_" + str(j) + "_level_" + str(level) + "_epoch_" +
                                        str(epoch) + ".jpg", enhanced)

                # Evaluate the model

                loss_mse_eval = 0
                loss_psnr_eval = 0
                loss_vgg_eval = 0
                loss_ssim_eval = 0

                generator.eval()
                with torch.no_grad():

                    test_iter = iter(test_loader)
                    for j in range(len(test_loader)):

                        x, y = next(test_iter)
                        x = x.to(device, non_blocking=True)
                        y = y.to(device, non_blocking=True)
                        enhanced = generator(x)

                        loss_mse_temp = MSE_loss(enhanced, y).item()

                        loss_mse_eval += loss_mse_temp
                        loss_psnr_eval += 20 * math.log10(1.0 / math.sqrt(loss_mse_temp))

                        if level < 2:
                            loss_ssim_eval += MS_SSIM(y, enhanced)

                        if level < 5:
                            enhanced_vgg_eval = VGG_19(normalize_batch(enhanced)).detach()
                            target_vgg_eval = VGG_19(normalize_batch(y)).detach()

                            loss_vgg_eval += MSE_loss(enhanced_vgg_eval, target_vgg_eval).item()

                loss_mse_eval = loss_mse_eval / TEST_SIZE
                loss_psnr_eval = loss_psnr_eval / TEST_SIZE
                loss_vgg_eval = loss_vgg_eval / TEST_SIZE
                loss_ssim_eval = loss_ssim_eval / TEST_SIZE

                if level < 2:
                    print("Epoch %d, mse: %.4f, psnr: %.4f, vgg: %.4f, ms-ssim: %.4f" % (epoch,
                            loss_mse_eval, loss_psnr_eval, loss_vgg_eval, loss_ssim_eval))
                elif level < 5:
                    print("Epoch %d, mse: %.4f, psnr: %.4f, vgg: %.4f" % (epoch,
                            loss_mse_eval, loss_psnr_eval, loss_vgg_eval))
                else:
                    print("Epoch %d, mse: %.4f, psnr: %.4f" % (epoch, loss_mse_eval, loss_psnr_eval))

                generator.train()


if __name__ == '__main__':
    train_model()

