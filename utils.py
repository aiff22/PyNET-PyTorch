# Copyright 2020 by Andrey Ignatov. All Rights Reserved.

import numpy as np
import sys
import os

NUM_DEFAULT_TRAIN_EPOCHS = [50, 25, 17, 17, 8, 8]


def process_command_args(arguments):

    # Specifying the default parameters

    level = 0
    batch_size = 50

    learning_rate = 5e-5

    restore_epoch = None
    num_train_epochs = None

    dataset_dir = 'raw_images/'

    for args in arguments:

        if args.startswith("level"):
            level = int(args.split("=")[1])

        if args.startswith("batch_size"):
            batch_size = int(args.split("=")[1])

        if args.startswith("learning_rate"):
            learning_rate = float(args.split("=")[1])

        if args.startswith("restore_epoch"):
            restore_epoch = int(args.split("=")[1])

        if args.startswith("num_train_epochs"):
            num_train_epochs = int(args.split("=")[1])

        if args.startswith("dataset_dir"):
            dataset_dir = args.split("=")[1]

    if restore_epoch is None and level < 5:
        restore_epoch = get_last_iter(level + 1)
        if restore_epoch == -1:
            print("Error: Cannot find any pre-trained models for PyNET's level " + str(level + 1) + ".")
            print("Aborting the training.")
            sys.exit()

    if num_train_epochs is None:
        num_train_epochs = NUM_DEFAULT_TRAIN_EPOCHS[level]

    print("The following parameters will be applied for CNN training:")

    print("Training level: " + str(level))
    print("Batch size: " + str(batch_size))
    print("Learning rate: " + str(learning_rate))
    print("Training epochs: " + str(num_train_epochs))
    print("Restore epoch: " + str(restore_epoch))
    print("Path to the dataset: " + dataset_dir)

    return level, batch_size, learning_rate, restore_epoch, num_train_epochs, dataset_dir


def process_test_model_args(arguments):

    level = 0
    restore_epoch = None

    dataset_dir = 'raw_images/'
    use_gpu = "true"

    orig_model = "false"

    for args in arguments:

        if args.startswith("level"):
            level = int(args.split("=")[1])

        if args.startswith("dataset_dir"):
            dataset_dir = args.split("=")[1]

        if args.startswith("restore_epoch"):
            restore_epoch = int(args.split("=")[1])

        if args.startswith("use_gpu"):
            use_gpu = args.split("=")[1]

        if args.startswith("orig"):
            orig_model = args.split("=")[1]

    if restore_epoch is None and orig_model == "false":
        restore_iter = get_last_iter(level)
        if restore_iter == -1:
            print("Error: Cannot find any pre-trained models for PyNET's level " + str(level) + ".")
            sys.exit()

    return level, restore_epoch, dataset_dir, use_gpu, orig_model


def get_last_iter(level):

    saved_models = [int((model_file.split("_")[-1]).split(".")[0])
                    for model_file in os.listdir("models/")
                    if model_file.startswith("pynet_level_" + str(level))]

    if len(saved_models) > 0:
        return np.max(saved_models)
    else:
        return -1


def normalize_batch(batch):
    # Normalize batch using ImageNet mean and std
    mean = batch.new_tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
    std = batch.new_tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
    return (batch - mean) / std
