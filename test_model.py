# Copyright 2020 by Andrey Ignatov. All Rights Reserved.

from scipy import misc
import numpy as np
import sys

from torch.utils.data import DataLoader
from torchvision import transforms
import torch

from load_data import LoadVisualData
from model import PyNET
import utils

to_image = transforms.Compose([transforms.ToPILImage()])

level, restore_epoch, dataset_dir, use_gpu, orig_model = utils.process_test_model_args(sys.argv)
dslr_scale = float(1) / (2 ** (level - 1))


def test_model():

    if use_gpu == "true":
        torch.backends.cudnn.deterministic = True
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # Creating dataset loaders

    visual_dataset = LoadVisualData(dataset_dir, 10, dslr_scale, level, full_resolution=True)
    visual_loader = DataLoader(dataset=visual_dataset, batch_size=1, shuffle=False, num_workers=0,
                               pin_memory=True, drop_last=False)

    # Creating and loading pre-trained PyNET model

    model = PyNET(level=level, instance_norm=True, instance_norm_level_1=True).to(device)
    model = torch.nn.DataParallel(model)

    if orig_model == "true":
        model.load_state_dict(torch.load("models/original/pynet_level_0.pth"), strict=True)
    else:
        model.load_state_dict(torch.load("models/pynet_level_" + str(level) +
                                             "_epoch_" + str(restore_epoch) + ".pth"), strict=True)

    model.half()
    model.eval()

    # Processing full-resolution RAW images

    with torch.no_grad():

        visual_iter = iter(visual_loader)
        for j in range(len(visual_loader)):

            print("Processing image " + str(j))

            torch.cuda.empty_cache()

            raw_image = next(visual_iter)
            raw_image = raw_image.to(device, dtype=torch.half)

            # Run inference

            enhanced = model(raw_image.detach())
            enhanced = np.asarray(to_image(torch.squeeze(enhanced.float().detach().cpu())))

            # Save the results as .png images

            misc.imsave("results/full-resolution/" + str(j) + "_level_" + str(level) +
                        "_epoch_" + str(restore_epoch) + ".png", enhanced)


if __name__ == '__main__':
    test_model()
