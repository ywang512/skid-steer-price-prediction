import os
import time
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import transforms

from preprocessing import csv2pickle
from modeling import init_datasets, train_model, save_model, load_model
from utils import getLogger, SkidSteerDataset, PriceModel
from evaluating import evaluate_model, evaluate_model_price


### Global Parameters
IMAGE_ROOT = "../data/images/"
RAW_FILEPATH = "../data/SkidSteer_2019-08.csv"
SCORE_FILEPATH = "./colorfulness/skid_steer_color_score.csv"
TRAIN_FILEPATH = "./SkidSteer_2019-08_clean_train.pickle"
VAL_FILEPATH = "./SkidSteer_2019-08_clean_val.pickle"
NUM_COLUMN_IDS = [2, 3, 4, 5, 7, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
ARRAY_COLUMN_ID = 22
MODEL_ROOT = "../models/"
MODEL_SAVE_NAME = "ftrs-images-text-sentiment"
MODEL_LOAD_NAME = None

### Global Toggles
PREPROCESS = True
TRAINING = True
EVALUATION_IDXS = [182, 292, 998, 788, 557, 886, 996, 348, 919, 118]
# EVALUATION_IDXS = list(range(1850))   # use this format to loop through the whole validation set for generating visualizations

### Model Parameters
RANDOM_SEED = 1
BATCH_SIZE = 16
NUM_EPOCHS = 300
LEARNING_RATE = 0.001
MOMENTUM = 0.9
SCHEDULER_REDUCE_ON_PLATEAU = True
SCHEDULER_STEP_SIZE = round(NUM_EPOCHS / 4)
HIDDEN_UNITS = [32]
FINE_TUNE = 2
TRANSFORM = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}


def main():
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    CURRENT_TIME = time.strftime("%Y-%m-%dT%H:%M", time.localtime())
    if MODEL_LOAD_NAME:
        model_save_path = os.path.join(MODEL_ROOT, MODEL_LOAD_NAME)
    else:
        model_save_path = os.path.join(MODEL_ROOT, CURRENT_TIME + "_" + MODEL_SAVE_NAME)
    try:
        os.makedirs(model_save_path)
    except:
        pass
    LOGGER = getLogger(name="main", model_save_path=model_save_path)
    if PREPROCESS:
        LOGGER.info("START preprocessing")
        scaler_dict = csv2pickle(raw_filepath=RAW_FILEPATH,
                                 score_filepath=SCORE_FILEPATH,
                                 image_root=IMAGE_ROOT,
                                 train_filepath=TRAIN_FILEPATH,
                                 val_filepath=VAL_FILEPATH,
                                 model_save_path=model_save_path,
                                 random_seed=RANDOM_SEED)

    LOGGER.info("START Initiating Datasets")
    LOGGER.info("Build Datasets")
    LOGGER.info("Build Dataloaders")
    datasets, dataloaders = init_datasets(train_filepath=TRAIN_FILEPATH,
                                          val_filepath=VAL_FILEPATH,
                                          image_root=IMAGE_ROOT,
                                          num_col_ids=NUM_COLUMN_IDS,
                                          array_col_id=ARRAY_COLUMN_ID,
                                          transform=TRANSFORM,
                                          batch_size=BATCH_SIZE)
    dataset_sizes = {x: len(datasets[x]) for x in ["train", "val"]}
    LOGGER.info("Sample size: Train %d" % dataset_sizes['train'])
    LOGGER.info("Sample size:  Val  %d" % dataset_sizes['val'])
    num_tabular_features = len(datasets["train"][0]["ftrs"])
    LOGGER.info("Numeric features count: %d" % len(NUM_COLUMN_IDS))
    LOGGER.info("Embedding features dimension: %d" % (num_tabular_features - len(NUM_COLUMN_IDS)))
    LOGGER.info("Features name: %s" % [nn for nn in datasets['train'].csv_file.columns[NUM_COLUMN_IDS+[ARRAY_COLUMN_ID]]])

    LOGGER.info("START Initiating Model")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    criterion = nn.MSELoss()
    if TRAINING:
        model = PriceModel(num_ftrs=num_tabular_features,
                           hidden_units=HIDDEN_UNITS,
                           fine_tune=FINE_TUNE).to(device)
        params = model.parameters()
        optimizer = optim.SGD(params, lr=LEARNING_RATE, momentum=MOMENTUM)
        if SCHEDULER_REDUCE_ON_PLATEAU:
            exp_lr_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, verbose=True)
        else:
            exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=SCHEDULER_STEP_SIZE)
        LOGGER.info("Model parameters: Training epochs = %d" % NUM_EPOCHS)
        LOGGER.info("Model parameters: Learning rate = %.3f" % LEARNING_RATE)
        LOGGER.info("Model parameters: Momentum = %.2f" % MOMENTUM)
        LOGGER.info("Model parameters: Scheduler step size = %d" % SCHEDULER_STEP_SIZE)
        LOGGER.info("Model parameters: FC layers = %s" % ([2048+num_tabular_features]+HIDDEN_UNITS+[1]))
        LOGGER.info("Model parameters: Fine tuning last %d layers" % FINE_TUNE)

        # train/save/read model
        LOGGER.info("START Training Model")
        model, all_records, best_records = train_model(dataloaders=dataloaders,
                                                       dataset_sizes=dataset_sizes,
                                                       model=model,
                                                       criterion=criterion,
                                                       optimizer=optimizer,
                                                       scheduler=exp_lr_scheduler,
                                                       num_epochs=NUM_EPOCHS,
                                                       device=device,
                                                       min_max_scaler=scaler_dict["winning_bid"],
                                                       model_save_path=model_save_path)
        model_load_path = model_save_path
        LOGGER.info("Save model to %s" % model_save_path)
        save_model(model_save_path=model_save_path,
                   model=model,
                   all_records=all_records,
                   best_records=best_records,
                   scaler_dict=scaler_dict)
    else:
        model_load_path = os.path.join(MODEL_ROOT, MODEL_LOAD_NAME)
        LOGGER.info("Load model from %s" % model_load_path)
        model, all_records, best_records, scaler_dict = load_model(model_load_path=model_load_path)

    # eval model
    LOGGER.info("START Evaluation")
    model = model.to("cpu")
    evaluate_model_price(model, device, datasets, "train", scaler_dict, TRAIN_FILEPATH, model_load_path)
    evaluate_model_price(model, device, datasets, "val", scaler_dict, VAL_FILEPATH, model_load_path)
    visualization_save_path = os.path.join(model_load_path, "eval_visualizations")
    try:
        os.makedirs(visualization_save_path)
    except:
        pass
    evaluate_model(model, dataset=datasets["val"], idxs=EVALUATION_IDXS, scaler=scaler_dict["winning_bid"], 
                   save_path=visualization_save_path, model_load_path=model_load_path)

    return None


if __name__ == "__main__":
    main()