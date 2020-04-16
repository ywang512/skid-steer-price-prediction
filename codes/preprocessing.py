"""Preprocess raw csv files into ready-to-train format

This module merge raw csv and score csv together, and then conduct preprocessing,
including FILTER columns, SELECT rows, IMPUTE nan, TRANSFORM numerical, NORMALIZE
numerical, INSERT new columns and SPLIT into train/val sets.

Todo:
    * add text cleaning part from Echo
    * add text embedding part from Echo
    * add MTurk features
"""

import os
import pickle
import logging
from collections import Counter

import numpy as np
import pandas as pd
from sklearn import preprocessing
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from utils import getLogger, getEmbedModel, getEmbed


### Local Parameters
COLUMN_NAMES = ['Unique_ID', 'Winning Bid', 'Hours Final', 'Age at Sale (bin)',
                'Bucket', 'Engine', 'Tires', 'Transmission', 'details remaining']
RENAME_SCHEMA = {
    'Unique_ID': "unique_id",
    'Hours Final': "hours_final",
    'Winning Bid': "winning_bid",
    'Age at Sale (bin)': "age_at_sale",
    'Bucket': "bucket",
    'Engine': "engine",
    'Tires': "tires",
    'Transmission': "transmission",
    'details remaining': "details_remaining",
    'socre': "colorfulness_score"
}

def csv2pickle(raw_filepath, score_filepath, image_root, train_filepath, val_filepath, model_save_path, random_seed):
    '''Join raw csv files into one master csv, with normalization and train-test-split.'''
    global LOGGER
    LOGGER = getLogger(name="Preprocess", model_save_path=model_save_path)
    LOGGER.info("Read raw csv")
    raw_df = pd.read_csv(raw_filepath, index_col=1)
    LOGGER.info("Read score csv")
    score_df = pd.read_csv(score_filepath)

    processed_df = clean_df(raw_df, score_df, image_root)
    impute_df(processed_df)
    transform_df(processed_df)
    scaler_dict = normalize_df(processed_df)
    add_binary_bucket(processed_df)
    part_sentiment_df(processed_df, ['bucket', 'engine', 'tires', 'transmission'])
    add_embeddings_df(processed_df, model_save_path)

    train_df, val_df = split_train_val(processed_df, random_seed)
    #train_df.to_csv(train_filepath)
    #val_df.to_csv(val_filepath)
    pickle.dump(train_df, open(train_filepath, "wb"))
    pickle.dump(val_df, open(val_filepath, "wb"))
    return scaler_dict


def clean_df(raw_df, score_df, image_root):
    '''Merge, rename columns and remove certain illegal rows.'''
    LOGGER.info("Rename columns")
    processed_df = raw_df.copy()
    processed_df['Unique_ID'] = processed_df[['Source', 'item#']].apply(lambda x: '_'.join(x), axis=1)
    processed_df = processed_df.filter(COLUMN_NAMES, axis=1)
    processed_df = pd.merge(processed_df, score_df, on='Unique_ID', how='inner')
    processed_df = processed_df.rename(columns=RENAME_SCHEMA)

    LOGGER.info("Remove rows with duplicate unique_id")
    duplicated_item = [item for item, count in Counter(processed_df["unique_id"]).items() if count > 1]
    processed_df = processed_df[~processed_df['unique_id'].isin(duplicated_item)]

    LOGGER.info("Remove rows with no matched images")
    image_item = [img_name.strip(".jpg") for img_name in os.listdir(image_root)]
    processed_df = processed_df[processed_df["unique_id"].isin(image_item)]

    LOGGER.info("Remove rows with corrupted images")
    processed_df = processed_df[processed_df['unique_id'] != "rbauction_10525632"]

    LOGGER.info("Remove comma in winning_bid")
    processed_df["winning_bid"] = processed_df["winning_bid"].str.replace(',', '').astype(int)
    return processed_df


def impute_df(data):
    '''Impute nan with median and new column of binary indicator.'''
    LOGGER.info("Impute column \"hours_final\"")
    data["hours_final"] = data["hours_final"].str.replace(",", "")
    data["hours_final"] = data["hours_final"].astype(float)
    data.insert(3, column="hours_final_nan", value=data["hours_final"].isna().astype(int))
    data.loc[data["hours_final"].isna(), "hours_final"] = data["hours_final"].median(skipna=True)

    LOGGER.info("Impute column \"age_at_sale\"")
    data["age_at_sale"] = data["age_at_sale"].astype(float)
    data.insert(5, column="age_at_sale_nan", value=data["age_at_sale"].isna().astype(int))
    data.loc[data["age_at_sale"].isna(), "age_at_sale"] = data["age_at_sale"].median(skipna=True)
    return None


def transform_df(data):
    '''Log transform the specific columns.'''
    LOGGER.info("Log-transform column \"winning_bid\"")
    data["winning_bid"] = np.log(data["winning_bid"])
    LOGGER.info("log-transofrm column \"hours_final\"")
    data["hours_final"] = np.log(data["hours_final"])
    return None


def normalize_df(data):
    '''Normalize specific columns with different scaler.'''
    scaler_dict = dict()
    LOGGER.info("Normalize (MinMaxScaler) column \"winning_bid\"")
    mm_scaler_price = preprocessing.MinMaxScaler((-1, 1))
    data["winning_bid"] = mm_scaler_price.fit_transform(data["winning_bid"].to_numpy().reshape(-1, 1))
    scaler_dict["winning_bid"] = mm_scaler_price

    LOGGER.info("Normalize (RobustScaler) column \"hours_final\"")
    rb_scaler_hour = preprocessing.RobustScaler()
    data["hours_final"] = rb_scaler_hour.fit_transform(np.array(data["hours_final"]).reshape(-1, 1))
    scaler_dict["hours_final"] = rb_scaler_hour

    LOGGER.info("Normalize (RobustScaler) column \"age_at_sale\"")
    rb_scaler_age = preprocessing.RobustScaler()
    data["age_at_sale"] = rb_scaler_age.fit_transform(np.array(data["age_at_sale"]).reshape(-1, 1))
    scaler_dict["age_at_sale"] = rb_scaler_age
    return scaler_dict


def add_binary_bucket(data):
    '''Add a column indicating whether it has bucket related words in bucket column.'''
    LOGGER.info("Add column \"bucket_binary\"")
    data.insert(7, column="bucket_binary", value=0)
    data.loc[
        ~data["bucket"].isna() &
        data["bucket"].str.contains("bucket", case=False) |
        data["bucket"].str.contains("bkt", case=False), "bucket_binary"
    ] = 1
    return None


def part_sentiment_df(data, parts):
    for part in parts:
        part_sentiment_name = part + "_sentiment"
        data[part_sentiment_name] = ""
        analyzer = SentimentIntensityAnalyzer()
        list_indices = np.where(~data[part].isnull())[0]
        for indice in list_indices:
            sentence = data[part].tolist()[indice]
            if len(sentence)>100:  #pick up only very strong signals
                vs = analyzer.polarity_scores(sentence)
                lb = vs['compound']
                if lb >= 0.05:
                    score = 1
                elif (lb > -0.05) and (lb < 0.05):
                    score = 0
                else:
                    score = -1
                data[part_sentiment_name][indice] = score
            else:
                data[part_sentiment_name][indice] = 0
        data[part_sentiment_name] = pd.to_numeric(data[part_sentiment_name]).fillna(0)


def add_embeddings_df(data, model_save_path):
    '''Compute, save and add embeddings of detais_remaining to the dataframe'''
    LOGGER.info("Producing text embeddings on details_remaining")
    sources = ['rbauct','big','iron','PW']
    list_embed = []
    list_index = []
    for source in sources:
        getEmbedModel(data, source, model_save_path)
        sentence_embedding, index = getEmbed(data, source, model_save_path)
        list_embed += sentence_embedding
        list_index += index
    index_nan = set(data.index) - set(list_index)

    LOGGER.info("Add %d column \"details_remaining_source\"" % len(sources))
    for source in sources:
        data.insert(len(data.columns), "details_remaining_source_"+source, 0)
    for ii, name_id in enumerate(data.unique_id):
        if name_id.startswith(sources[0]):
            data.iloc[ii, -4] = 1
        elif name_id.startswith(sources[1]):
            data.iloc[ii, -3] = 1
        elif name_id.startswith(sources[2]):
            data.iloc[ii, -2] = 1
        elif name_id.startswith(sources[3]):
            data.iloc[ii, -1] = 1

    LOGGER.info("Add column \"details_remaining_nan\"")
    data.insert(len(data.columns), "details_remaining_nan", 0)
    data.loc[data["details_remaining"].isna(), "details_remaining_nan"] = 1

    LOGGER.info("Add column \"details_remaining_embedding\"")
    embeds_median = np.median(np.array(list_embed), axis=0)
    embeds = {index:embed for index, embed in zip(list_index, list_embed)}
    embeds_list = [embeds[ii] if ii not in index_nan else embeds_median for ii in data.index]
    data.insert(len(data.columns), "details_remaining_embedding", embeds_list)
    return None


def split_train_val(data, random_seed, split=[0.7, 0.3]):
    '''Split preprocessed data into train and val sets.'''
    LOGGER.info("Split into train and val with %s" % str(split))
    np.random.seed(random_seed)
    split0 = round(data.shape[0] * split[0])
    # split1 = round(data.shape[0] * (split[0] + split[1]))
    data = data.sample(frac=1)
    df_train = data.iloc[:split0]
    df_val = data.iloc[split0:]
    return df_train, df_val
