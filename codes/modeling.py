'''
TODO:
    * docstring
    * logging
'''

import os
import time
import copy
import pickle
import logging

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, models, utils

from utils import getLogger, SkidSteerDataset, PriceModel, compute_price_loss


def init_datasets(train_filepath, val_filepath, image_root, num_col_ids, array_col_id, transform, batch_size):
    '''Initiate datasets.'''
    csv_file = {"train": train_filepath,
                "val": val_filepath}
    datasets = {x: SkidSteerDataset(csv_file=csv_file[x],
                                    img_root=image_root,
                                    num_col_ids=num_col_ids,
                                    array_col_id=array_col_id,
                                    transform=transform[x])
                for x in ["train", "val"]}
    dataloaders = {x: DataLoader(datasets[x], 
                                 batch_size=batch_size, 
                                 shuffle=True, 
                                 num_workers=4)
                   for x in ["train", "val"]}
    return datasets, dataloaders


def train_model(dataloaders, dataset_sizes, model, criterion, optimizer, 
                scheduler, num_epochs, device, min_max_scaler, model_save_path):
    '''Train models.'''
    LOGGER = getLogger(name="Train", model_save_path=model_save_path)

    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_epoch = None
    best_loss = float("Inf")
    best_mae = float("Inf")
    best_maep = float("Inf")
    best_mae_list = None
    best_maep_list = None
    all_loss = {x: [] for x in ['train', 'val']}
    all_mae = {x: [] for x in ['train', 'val']}
    all_maep = {x: [] for x in ['train', 'val']}

    for epoch in range(num_epochs):
        LOGGER.info('Epoch {}/{}'.format(epoch + 1, num_epochs))

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode
            running_loss = 0.0
            running_mae = 0.0
            running_maep = 0.0
            running_mae_list = []
            running_maep_list = []
            
            # Iterate over data.
            for items in dataloaders[phase]:
                prices = items["price"].to(device)
                images = items["image"].to(device)
                ftrs = items["ftrs"].to(device)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(images, ftrs)
                    loss = criterion(outputs.squeeze(), prices)
                    mae, mae_np, maep, maep_np = compute_price_loss(outputs, prices, min_max_scaler)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                running_loss += loss.item() * images.size(0)
                running_mae += mae
                running_maep += maep
                running_mae_list += list(mae_np.flatten())
                running_maep_list += list(maep_np.flatten())
            if phase == 'train':
                if type(scheduler) == torch.optim.lr_scheduler.ReduceLROnPlateau:
                    scheduler.step(running_loss)
                else:
                    scheduler.step()
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_mae = running_mae / dataset_sizes[phase]
            epoch_maep = running_maep / dataset_sizes[phase]
            all_loss[phase].append(epoch_loss)
            all_mae[phase].append(epoch_mae)
            all_maep[phase].append(epoch_maep)
            LOGGER.info('{:^5} Loss: {:.4f}   MAE: {:.2f}   MAEP: {:.2f} %'.format(phase.upper(), 
                                                                                   epoch_loss,
                                                                                   epoch_mae,
                                                                                   100*epoch_maep))
            
            # deep copy the model
            if phase == 'val' and epoch_loss < best_loss:
                best_epoch = epoch + 1
                best_loss = epoch_loss
                best_mae = epoch_mae
                best_maep = epoch_maep
                best_mae_list = running_mae_list
                best_maep_list = running_maep_list
                best_model_wts = copy.deepcopy(model.state_dict())

    time_elapsed = time.time() - since
    LOGGER.info('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    LOGGER.info('Best Epoch (in terms of VAL Loss): {}'.format(best_epoch))
    LOGGER.info('Best VAL Loss: {:.4f}'.format(best_loss))
    LOGGER.info('Best VAL MAE: {:.2f}'.format(best_mae))
    LOGGER.info('Best VAL MAEP: {:.2f}'.format(100*best_maep))

    # load best model weights
    LOGGER.info("Load the model weights at the best epoch {}".format(best_epoch))
    model.load_state_dict(best_model_wts)
    all_records = {
        "loss": all_loss,
        "mae": all_mae,
        "maep": all_maep
    }
    best_records = {
        "epoch": best_epoch,
        "loss": best_loss,
        "mae": best_mae,
        "maep": best_maep,
        "mae_list": best_mae_list,
        "maep_list": best_maep_list
    }
    return model, all_records, best_records


def save_model(model_save_path, model, all_records, best_records, scaler_dict):
    '''Save model and training statistics to the given path.'''
    model_path = os.path.join(model_save_path, "model.pt")
    all_records_path = os.path.join(model_save_path, "all_records.pickle")
    best_records_path = os.path.join(model_save_path, "best_records.pickle")
    scaler_dict_path = os.path.join(model_save_path, "scaler_dict.pickle")
    torch.save(model, model_path)
    pickle.dump(all_records, open(all_records_path, "wb"))
    pickle.dump(best_records, open(best_records_path, "wb"))
    pickle.dump(scaler_dict, open(scaler_dict_path, "wb"))
    return None


def load_model(model_load_path):
    '''Load model and training statistics from the given path to cpu.'''
    model_path = os.path.join(model_load_path, "model.pt")
    all_records_path = os.path.join(model_load_path, "all_records.pickle")
    best_records_path = os.path.join(model_load_path, "best_records.pickle")
    scaler_dict_path = os.path.join(model_load_path, "scaler_dict.pickle")
    model = torch.load(model_path, map_location=torch.device('cpu'))
    all_records = pickle.load(open(all_records_path, "rb"))
    best_records = pickle.load(open(best_records_path, "rb"))
    scaler_dict = pickle.load(open(scaler_dict_path, "rb"))
    return model, all_records, best_records, scaler_dict