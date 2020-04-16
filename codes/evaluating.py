import os
import pickle
import logging
from copy import deepcopy

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.ndimage.interpolation import zoom
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models, utils

from utils import getLogger, norm2price


class CamExtractor():
    """Extracts CAM features from the model."""
    
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None

    def save_gradient(self, grad):
        self.gradients = grad

    def forward_pass_on_convolutions(self, x):
        """Does a forward pass on convolutions, hooks the function at given layer."""
        conv_output = None
        for module_pos, module in self.model.cnn._modules.items():
            x = module(x)  # Forward
            if str(module_pos) == self.target_layer:
                x.register_hook(self.save_gradient)
                conv_output = x  # Save the convolution output on that layer
        return conv_output, x

    def forward_pass(self, image, ftrs):
        """Does a full forward pass on the model."""
        conv_output, x = self.forward_pass_on_convolutions(image)
        x = x.view(x.size(0), -1)  # Flatten
        # Forward pass on the classifier
        x = torch.cat([x, ftrs], dim=1)
        x = self.model.fc_layers(x)
        return conv_output, x


class GradCam():
    """Produces class (price) activation map."""
    def __init__(self, model, target_layer):
        self.model = model
        self.model.eval()
        # Define extractor
        self.extractor = CamExtractor(self.model, target_layer)

    def generate_cam(self, input_image, input_ftrs, target_price=None):
        '''Generate CAM for the inputs.'''
        conv_output, model_output = self.extractor.forward_pass(input_image, input_ftrs)
        if target_price is None:
            target_price = model_output
        target_price_array = torch.FloatTensor(1, 1).zero_()
        target_price_array[0][0] = target_price
        self.model.cnn.zero_grad()
        self.model.fc_layers.zero_grad()

        model_output.backward(gradient=target_price_array, retain_graph=True)
        guided_gradients = self.extractor.gradients.data.numpy()[0]  # (2048, 7, 7)
        weights = np.mean(guided_gradients, axis=(1, 2)).reshape(-1, 1, 1)  # Take averages for each gradient  # (2048, 1, 1)
        features_map = conv_output.data.numpy()[0]  # (2048, 7, 7)

        cam = np.sum(features_map * weights, axis=0) + 1
        cam = np.maximum(cam, 0)
        cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam))  # Normalize between 0-1
        cam = np.uint8(cam * 255)  # Scale between 0-255 to visualize
        cam = zoom(cam, np.array(input_image[0].shape[1:])/np.array(cam.shape))
        # cam = np.uint8(Image.fromarray(cam).resize((input_image.shape[2],
        #                input_image.shape[3]), Image.ANTIALIAS))/255
        return cam, model_output


class GuidedBackprop():
    """Produces gradients generated with guided back propagation from the given image."""
    
    def __init__(self, model):
        self.model = model
        self.gradients = None
        self.forward_relu_outputs = []
        # Put model in evaluation mode
        for param in self.model.parameters():
            param.requires_grad = True
        self.model.eval()
        self.update_relus()

    def update_relus(self):
        """
        Updates relu activation functions so that
            1- stores output in forward pass
            2- imputes zero for gradient values that are less than zero
        """
        def relu_backward_hook_function(module, grad_in, grad_out):
            """If there is a negative gradient, change it to zero."""
            # Get last forward output
            corresponding_forward_output = self.forward_relu_outputs[-1]
            corresponding_forward_output[corresponding_forward_output > 0] = 1
            modified_grad_out = corresponding_forward_output * torch.clamp(grad_in[0], min=0.0)
            del self.forward_relu_outputs[-1]  # Remove last forward output
            return (modified_grad_out,)

        def relu_forward_hook_function(module, ten_in, ten_out):
            """Store results of forward pass."""
            self.forward_relu_outputs.append(ten_out)

        # Loop through layers, hook up ReLUs
        for module in list(self.model.cnn.modules()):
            if isinstance(module, nn.ReLU):
                module.register_backward_hook(relu_backward_hook_function)
                module.register_forward_hook(relu_forward_hook_function)

    def generate_gradients(self, input_image, input_ftrs, target_price):
        '''Generate gradients of price with respect to input image.'''
        input_image.requires_grad = True
        model_output = self.model(input_image, input_ftrs)
        self.model.zero_grad()
        target_price_array = torch.FloatTensor(1, 1).zero_()
        target_price_array[0][0] = target_price
        model_output.backward(gradient=target_price_array)
        image_gradients = input_image.grad.squeeze()
        #gradients_as_arr = self.gradients.data.numpy()[0]
        #gradients_as_arr = (gradients_as_arr - gradients_as_arr.min()) / (gradients_as_arr.max() - gradients_as_arr.min())
        # norm1 - [0..1], 1 filter
        image_gradients_norm1 = image_gradients.data.numpy().max(axis=0)
        image_gradients_norm1 = (image_gradients_norm1 - image_gradients_norm1.min()) / (image_gradients_norm1.max() - image_gradients_norm1.min())
        # norm2 - [-1..1], 1 filter
        image_gradients_norm2 = image_gradients.data.numpy().max(axis=0)
        image_gradients_norm2 = image_gradients_norm2*(1/max(image_gradients_norm2.max(), abs(image_gradients_norm2.min())))
        # norm3 - [0..1], 3 filters
        image_gradients_norm3 = image_gradients.data.numpy()
        image_gradients_norm3 = (image_gradients_norm3 - image_gradients_norm3.min()) / (image_gradients_norm3.max() - image_gradients_norm3.min())
        return (image_gradients_norm1, image_gradients_norm2, image_gradients_norm3), model_output


def make_CAM_plots(original_image, cam, original_price, predicted_price, unique_id, save_path):
    '''Make plots for Classification Attention Maps visualization.'''
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10.1, 5), gridspec_kw={'width_ratios': [5, 5, 0.1]})
    ax1.imshow(invTrans(original_image.squeeze()).detach().permute(1, 2, 0))
    ax1.set_title("Original")
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax2.imshow(cam, cmap="RdYlGn")
    ax2.set_title("CAM")
    ax2.set_xticks([])
    ax2.set_yticks([])
    cmap = mpl.cm.RdYlGn
    norm = mpl.colors.Normalize(vmin=-1, vmax=1)
    cb1 = mpl.colorbar.ColorbarBase(ax3, cmap=cmap, norm=norm, orientation='vertical')
    plt.suptitle("Real price: %d   ~   Predicted: %d" % (original_price, np.round(predicted_price)))
    image_path = os.path.join(save_path, str(unique_id) + "-cam.png")
    plt.savefig(image_path)
    return None


def make_GB_plots(guided_grads, original_price, predicted_price, unique_id, save_path):
    '''Make plots for all visualization.'''
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    (ax1, ax2, ax3, ax4) = axs.flatten()
    ax1.imshow(guided_grads[0], cmap="gray")
    ax1.set_title("Monochromatic")
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax2.imshow(guided_grads[2].transpose(1, 2, 0))
    ax2.set_title("Colorful")
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax3.imshow(np.minimum(guided_grads[1], 0), cmap="RdYlGn", vmin=guided_grads[1].min(), vmax=-guided_grads[1].min())
    ax3.set_title("Negative")
    ax3.set_xticks([])
    ax3.set_yticks([])
    ax4.imshow(np.maximum(guided_grads[1], 0), cmap="RdYlGn", vmin=-1, vmax=1)
    ax4.set_title("Positive")
    ax4.set_xticks([])
    ax4.set_yticks([])
    fig.suptitle("  Real price: %d   ~   Predicted: %d" % (original_price, np.round(predicted_price)), size=16)
    fig.subplots_adjust(top=0.94)
    image_path = os.path.join(save_path, str(unique_id) + "-gb.png")
    plt.savefig(image_path)
    return None


def make_all_plots(original_image, cam, guided_grads, original_price, predicted_price, unique_id, save_path):
    '''Make plots for all visualization.'''
    fig, axs = plt.subplots(3, 2, figsize=(10, 15))
    (ax1, ax2, ax3, ax4, ax5, ax6) = axs.flatten()
    ax1.imshow(invTrans(original_image.squeeze()).permute(1, 2, 0).data.numpy())
    ax1.set_title("Original")
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax2.imshow(cam, cmap="RdYlGn")
    ax2.set_title("Attention")
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax3.imshow(guided_grads[0], cmap="gray")
    ax3.set_title("Monochromatic")
    ax3.set_xticks([])
    ax3.set_yticks([])
    ax4.imshow(guided_grads[2].transpose(1, 2, 0))
    ax4.set_title("Colorful")
    ax4.set_xticks([])
    ax4.set_yticks([])
    ax5.imshow(np.minimum(guided_grads[1], 0), cmap="RdYlGn", vmin=guided_grads[1].min(), vmax=-guided_grads[1].min())
    ax5.set_title("Negative")
    ax5.set_xticks([])
    ax5.set_yticks([])
    ax6.imshow(np.maximum(guided_grads[1], 0), cmap="RdYlGn", vmin=-1, vmax=1)
    ax6.set_title("Positive")
    ax6.set_xticks([])
    ax6.set_yticks([])
    fig.suptitle("  Real price: %d   ~   Predicted: %d" % (original_price, np.round(predicted_price)), size=16)
    fig.subplots_adjust(top=0.94)
    image_path = os.path.join(save_path, str(unique_id) + "-all.png")
    plt.savefig(image_path)
    return None


def evaluate_model_price(model, device, datasets, phase, scaler_dict, pickle_filepath, model_load_path):
    '''Output orginal and predicted price to a new csv file.'''
    LOGGER = getLogger(name="Evaluate", model_save_path=model_load_path)
    LOGGER.info("Predicting price on %s dataset" % phase)
    model = deepcopy(model).to(device)
    model.eval()
    ori_price = []
    pre_price = []
    for ii in range(len(datasets[phase])):
        price = datasets[phase][ii]["price"]
        image = datasets[phase][ii]["image"].unsqueeze(dim=0).to(device)
        ftrs = datasets[phase][ii]["ftrs"].unsqueeze(dim=0).to(device)
        outputs = model(image, ftrs)
        pre_price.append(norm2price(outputs, scaler_dict["winning_bid"]).flatten()[0])
        ori_price.append(norm2price(torch.tensor(price), scaler_dict["winning_bid"]).flatten()[0])

    df = pickle.load(open(pickle_filepath, "rb"))
    df.insert(len(df.columns), "original_price", np.array(ori_price))
    df.insert(len(df.columns), "predicted_price", np.array(pre_price))
    pickle_savepath = os.path.join(model_load_path, "results_"+phase+".csv")
    pickle.dump(df, open(pickle_savepath, "wb"))
    LOGGER.info("Save result pickled csv to %s" % pickle_savepath)
    return None


def evaluate_network_one(model, dataset, idx, scaler, save_path, LOGGER):
    '''Evaluate model performance on one image.'''
    unique_id = dataset[idx]["unique_id"]
    price = torch.tensor(dataset[idx]["price"])
    image = dataset[idx]["image"].unsqueeze(dim=0)
    ftrs = dataset[idx]["ftrs"].unsqueeze(dim=0)

    gradcam = GradCam(deepcopy(model), target_layer="layer4")
    guidedbackprop = GuidedBackprop(deepcopy(model))
    cam, model_price1 = gradcam.generate_cam(image, ftrs, target_price=price)
    guided_grads, model_price2 = guidedbackprop.generate_gradients(image, ftrs, target_price=price)
    if model_price1 != model_price2:
        LOGGER.warning("unmatched price at unique_id = %s" % unique_id)

    original_price = norm2price(price, scaler).flatten()
    model_price = norm2price(model_price1, scaler).squeeze()
    # original_price = price
    # model_price = float(model_price1)
    make_CAM_plots(image, cam, original_price, model_price, unique_id, save_path)
    make_GB_plots(guided_grads, original_price, model_price, unique_id, save_path)
    # make_all_plots(image, cam, guided_grads, original_price, model_price, idx, save_path)
    LOGGER.info("Visualization unique_id = %s (idx = %d) saved to %s" % (unique_id, idx, save_path))
    return None


def evaluate_model(model, dataset, idxs, scaler, save_path, model_load_path):
    '''Generate CAM and GB maps for specific images.'''
    LOGGER = getLogger(name="Evaluate", model_save_path=model_load_path)
    global invTrans 
    invTrans = transforms.Compose([
        transforms.Normalize(mean=[ 0., 0., 0. ], std=[ 1/0.229, 1/0.224, 1/0.225 ]),
        transforms.Normalize(mean=[ -0.485, -0.456, -0.406 ], std=[ 1., 1., 1. ])
    ])

    LOGGER.info("Evaluate on ids = %s" % idxs)
    for idx in idxs:
        evaluate_network_one(model, dataset, idx, scaler, save_path, LOGGER)
    LOGGER.info("Evaluation finish")
    return None


def main():
    '''Direct call to evalute the model.'''
    invTrans = transforms.Compose([
        transforms.Normalize(mean=[ 0., 0., 0. ], std=[ 1/0.229, 1/0.224, 1/0.225 ]),
        transforms.Normalize(mean=[ -0.485, -0.456, -0.406 ], std=[ 1., 1., 1. ])
    ])
    return None


if __name__ == "__main__":
    main()