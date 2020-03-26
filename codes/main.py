import os
import time
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import transforms

from preprocessing import csv2csv
from modeling import init_datasets, train_model, save_model, load_model
from utils import SkidSteerDataset, PriceModel
from evaluating import evaluate_model




### Logging
FORMAT = '%(asctime)-15s %(levelname)s [%(name)s] %(message)s'
logging.basicConfig(format=FORMAT)
LOGGER = logging.getLogger('main')
LOGGER.setLevel("INFO")
CURRENT_TIME = time.strftime("%Y-%m-%dT%H:%M", time.localtime())


### Global Parameters
IMAGE_ROOT = "../data/images/"
RAW_FILEPATH = "../data/SkidSteer_2019-08.csv"
SCORE_FILEPATH = "./colorfulness/skid_steer_color_score.csv"
TRAIN_FILEPATH = "./SkidSteer_2019-08_clean_train.csv"
VAL_FILEPATH = "./SkidSteer_2019-08_clean_val.csv"
COLUMN_IDS = [2, 3, 4, 5, 7, 11]
MODEL_ROOT = "../models/"
MODEL_SAVE_NAME = "test"
MODEL_LOAD_NAME = "2020-03-17T01:31_test"

### Global Toggles
PREPROCESS = True
TRAINING = True
EVALUATION_IDXS = [159, 787]

### Model Parameters
RANDOM_SEED = 1
BATCH_SIZE = 16
NUM_EPOCHS = 200
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
    if PREPROCESS:
        LOGGER.info("START preprocessing")
        scaler_dict = csv2csv(raw_filepath=RAW_FILEPATH,
                              score_filepath=SCORE_FILEPATH,
                              image_root=IMAGE_ROOT,
                              train_filepath=TRAIN_FILEPATH,
                              val_filepath=VAL_FILEPATH,
                              random_seed=RANDOM_SEED)

    LOGGER.info("START Initiating Datasets")
    LOGGER.info("Build Datasets")
    LOGGER.info("Build Dataloaders")
    datasets, dataloaders = init_datasets(train_filepath=TRAIN_FILEPATH,
                                          val_filepath=VAL_FILEPATH,
                                          image_root=IMAGE_ROOT,
                                          col_ids=COLUMN_IDS,
                                          transform=TRANSFORM,
                                          batch_size=BATCH_SIZE)
    dataset_sizes = {x: len(datasets[x]) for x in ["train", "val"]}
    LOGGER.info("Sample size: Train %d" % dataset_sizes['train'])
    LOGGER.info("Sample size:  Val  %d" % dataset_sizes['val'])
    num_tabular_features = len(datasets["train"][0]["ftrs"])
    LOGGER.info("Features count: %d" % num_tabular_features)
    LOGGER.info("Features name: %s" % [nn for nn in datasets['train'].csv_file.columns[COLUMN_IDS]])

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
                                                       min_max_scaler=scaler_dict["winning_bid"])
        model_save_path = os.path.join(MODEL_ROOT, CURRENT_TIME + "_" + MODEL_SAVE_NAME)
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
    # evaluate_model(model, idxs=EVALUATION_IDXS, scaler=scaler_dict["winning_bid"], save_path=model_load_path)
    evaluate_model(model, dataset=datasets["val"], idxs=EVALUATION_IDXS, scaler=None, save_path=model_load_path)

    return None


if __name__ == "__main__":
    main()