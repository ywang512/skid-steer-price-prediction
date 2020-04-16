# Price Prediction Model for Skid Steer Data

This repo contains all the necessary codes for building this model. Sensitive data are either not included or desensitized. Due to the storage limit of github repo, large files has been put into Google drive for future use.

Please refer to this white paper and slides for a better understanding of this model. This README file will focus on how to use this tool.

## How to Use

To train a new model or evaluate an exsited model, simply use python to run `main.py`. Make sure it's in the same folder with `preprocessing.py`, `modeling.py`, `evaluating.py` and `utils.py`.

To calculate the colorfulness score for a new dataset, please refer to the scripts in `codes/colorfulness/`.


## Main Arguments Explanation

These arguments determine which file to read and where to save outputs.

 - `IMAGE_ROOT`: File path to the root directory of images.

 - `RAW_FILEPATH`: File path to the original csv tabular dataset.

 - `SCORE_FILEPATH`: File path to the colorfulness score csv.

 - `TRAIN_FILEPATH`: File path to save the preprocessed training dataset.

 - `VAL_FILEPATH`: File path to save the preprocessed validation dataset.

 - `NUM_COLUMN_IDS`: A list of column indexes of numerical data in preprocessed training dataset to be put into model training as predictors.

 - `ARRAY_COLUMN_ID`: An integer of column index of the preprocessed text embeddings to be put into model training as predictors.

 - `MODEL_ROOT`: File path to the root directory of all models.

 - `MODEL_SAVE_NAME`: A folder in `MODEL_ROOT` to save the newly trained model outputs.

 - `MODEL_LOAD_NAME`: A folder in `MODEL_ROOT` to load a previously trained model parameters.


These arguments determine the purpose of running `main.py`.

 - `PREPROCESS`: Whether to preprocess the data or not. If `True`, new training and validation dataset will be created.

 - `TRAINING`: Whether to train a new model or evaluate an exsisted model. If `True`, new model will be trained and saved to `MODEL_SAVE_NAME`. Otherwise an exsisted model will be read from `MODEL_LOAD_NAME`.

 - `EVALUATION_IDXS`: A list of validation ids (in validation dataset) to be evaluated in this run.


These arguments determine the hyperparameters of the model. 

 - `RANDOM_SEED`: A random seed used to train-test-split dataset and initialize model weights.

 - `BATCH_SIZE`: The number of different data points in each batch.

 - `NUM_EPOCHS`: The numebr of epochs the model will be trained for.

 - `LEARNING_RATE`: The learning rate of the optimizer (stochastic gradient descent).

 - `MOMENTUM`: The momentum of the optimizer (stochastic gradient descent).

 - `SCHEDULER_REDUCE_ON_PLATEAU`: Whether to use this scheduler to determine the trigger of reducing learning rate. If `True`, the model will reduce the current learning rate whenever the validation performance reaches a plateau (no more improve). Otherwise, `SCHEDULER_STEP_SIZE` will be used to determine how often the learning rate will be reduced.

 - `SCHEDULER_STEP_SIZE`: The number of epochs 

 - `HIDDEN_UNITS`: The struture and numebrs of hidden nodes of fully-connected layers after the concatenation of image array, text array and tabular array.

 - `FINE_TUNE`: An integer showing the depth of fine tuning convolutional blocks. If X, it means the last X covolutional blocks of pretrained model will be fine-tuned.

 - `TRANSFORM`: A `torchvision.transforms` object. Make sure to resize and normalize the images in order to meet the input requirements of the specific pre-trained model (ResNet152).


## Output Explanation

Some trained models are included in the `/models/` path. By default all the model will be named by "starting time" and `MODEL_SAVE_NAME`. `2020-04-15T03:01_ftrs-images` contains a model trained by using tabular features and images. `2020-04-01T20:09_ftrs-images-text` contains a model trained by using tabular features, images and comment embeddings. `2020-04-11T20:08_ftrs-images-text-sentiment` contains a model trained by using tabular features, images, comment embeddings and comment sentiment.

Within each model folder, there are many different files will be generated for each training run.

 - `eval_visualizations` contains all the output visualizations for the specific images. All output visualizations are named by combining "unique id" and the type of visualization. `cam` stands for classificaiton attention mapping and `gb` stands for guided backproporgation (saliency) mapping.

 - `all_records.pickle` is a pickle file containing a dictionary of all statistics (loss, MAE and MAEP) for each epochs.

 - `best_records.pickle` is a pickle file containing a dictionary of all statistics (loss, MAE, MAEP, individual MAE and individual MAEP) for the epoch with best validation performance.

 - `details_text_big.csv` contains cleaned details text for data from bigiron source.

 - `details_text_iron.csv` contains cleaned details text for data from ironplanet source.

 - `details_text_PW.csv` contains cleaned details text for data from Purple Wave source.

 - `details_text_rbauct.csv` contains cleaned details text for data from rbauction source.

 - `details_text_skipgram_model_big.bin` contains the text embeddings model for details text from bigiron source.

 - `details_text_skipgram_model_iron.bin` contains the text embeddings model for details text from ironplanet source.

 - `details_text_skipgram_model_PW.bin` contains the text embeddings model for details text from Purple Wave source.

 - `details_text_skipgram_model_rbauct.bin` contains the text embeddings model for details text from rbauction source.

 - `model.pt` contains all the model weights for our price prediction model.

 - `results_train.csv` is a csv file with true price and predicted price of training dataset.

 - `results_val.csv` is a csv file with true price and predicted price of validation dataset.

 - `scaler_dict.pickle` is a pickle file containing a dictionary of all scaler used to preprocess numerical data, so that reverse transform is possible.

 - `stdout_console.log` is the log file of all the manipulation conducted on this model in `main.py`


## Future Work

 - Integrate calculating colorfulness score into the preprocessing pipeline.

 - Gather more human annotations (such as MTurk) and integrate them into this model. (See Discussion section in this paper)

 - Try adding a few fully-connected layer right after getting the 2048 image features to downweight the effect of images. (See Discussion section in this paper)

 - Evaluate more image visualizations and try to explore a more consistent interpretation for them. (See Discussion section in this paper)