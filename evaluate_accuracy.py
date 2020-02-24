# Copyright 2020 by Andrey Ignatov. All Rights Reserved.

from torch.utils.data import DataLoader
import numpy as np
import torch
import math

from load_data import LoadData
from msssim import MSSSIM
from model import PyNET

np.random.seed(0)
torch.manual_seed(0)

# Path to the dataset:
dataset_dir = 'raw_images/'
TEST_SIZE = 1204


def evaluate_accuracy():

    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda")

    print("CUDA visible devices: " + str(torch.cuda.device_count()))
    print("CUDA Device Name: " + str(torch.cuda.get_device_name(device)))

    # Create test dataset loader
    test_dataset = LoadData(dataset_dir, TEST_SIZE, 2.0, test=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=1,
                             pin_memory=True, drop_last=False)

    # Define the model architecture and restore it from the .pth file, e.g.:

    model = PyNET(level=0, instance_norm=True, instance_norm_level_1=True).to(device)
    model = torch.nn.DataParallel(model)
    model.load_state_dict(torch.load("models/original/pynet_level_0.pth"), strict=True)

    # Define the losses

    MSE_loss = torch.nn.MSELoss()
    MS_SSIM = MSSSIM()

    loss_psnr = 0.0
    loss_msssim = 0.0

    model.eval()
    with torch.no_grad():

        test_iter = iter(test_loader)
        for j in range(len(test_loader)):

            x, y = next(test_iter)
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            # Process raw images with your model:
            enhanced = model(x)

            # Compute losses
            loss_mse_temp = MSE_loss(enhanced, y).item()
            loss_psnr += 20 * math.log10(1.0 / math.sqrt(loss_mse_temp))

            loss_msssim += MS_SSIM(y, enhanced).detach().cpu().numpy()

    loss_psnr = loss_psnr / TEST_SIZE
    loss_msssim = loss_msssim / TEST_SIZE

    output_logs = "PSNR: %.4g, MS-SSIM: %.4g\n" % (loss_psnr, loss_msssim)
    print(output_logs)


if __name__ == '__main__':
    evaluate_accuracy()

