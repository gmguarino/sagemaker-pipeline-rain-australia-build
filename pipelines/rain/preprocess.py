import logging
import sys

import argparse
import os
import pathlib
import json
import joblib
import tarfile

import boto3
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from io import StringIO

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))


def merge_two_dicts(x, y):
    """Merges two dicts, returning a new copy."""
    z = x.copy()
    z.update(y)
    return z


def cyclical_encode(data, col, max_val):
    data[col + '_sin'] = np.sin(2 * np.pi * data[col]/max_val)
    data[col + '_cos'] = np.cos(2 * np.pi * data[col]/max_val)
    return data


if __name__ == "__main__":

    logger.debug("Starting preprocessing.")
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--input-data", type=str, required=True)
    # args = parser.parse_args()

    base_dir = "/opt/ml/processing"
    # pathlib.Path(f"{base_dir}/data").mkdir(parents=True, exist_ok=True)
    # input_data = args.input_data
    # bucket = input_data.split("/")[2]
    # key = "/".join(input_data.split("/")[3:])


    # logger.debug("Downloading data from bucket: %s, key: %s", bucket, key)
    # fn = f"{base_dir}/data/rain-au-dataset.csv"
    fn = f"{base_dir}/input/weatherAUS.csv"
    # s3 = boto3.resource("s3")
    # s3.Bucket(bucket).download_file(key, fn)
    

    logger.debug("Reading downloaded data.")
    df = pd.read_csv(fn)
    os.unlink(fn)

    logger.debug("Parsing dates as Datetime, and cyclically encoding day and month")
    df['Date']= pd.to_datetime(df["Date"])
    df['year'] = df["Date"].dt.year
    df['month'] = df["Date"].dt.month
    df = cyclical_encode(df, 'month', 12)

    df['day'] = df["Date"].dt.day
    df = cyclical_encode(df, 'day', 31)

    logger.debug("Handle missing categorical data")
    # Get list of categorical data
    s = (df.dtypes == "object")
    object_cols = list(s[s].index)
    for i in object_cols:
        # Fill missing data with mode
        df[i].fillna(   df[i].mode()[0], inplace=True)

    logger.debug("Handle missing numerical data")
    # Get list of numerical data
    t = (df.dtypes == "float64")
    num_cols = list(t[t].index)
    for i in num_cols:
        # Fill in missing data w/ median
        df[i].fillna(df[i].median(), inplace=True)


    logger.debug("Encode Categorical Data")
    label_encoder = LabelEncoder()
    for i in object_cols:
        df[i] = label_encoder.fit_transform(df[i])

    logger.debug("Dropping unnecessary columns, scaling and separating target")
    features = df.drop(['RainTomorrow', 'Date','day', 'month'], axis=1) # dropping target and extra columns

    target = df['RainTomorrow']

    #Set up a standard scaler for the features
    col_names = list(features.columns)
    s_scaler = StandardScaler()
    features = s_scaler.fit_transform(features)
    features = pd.DataFrame(features, columns=col_names) 


    logger.debug("Drop outliers")
    # reinsert target after scaling
    features["RainTomorrow"] = target

    #Dropping with outlier

    features = features[(features["MinTemp"]<2.3)&(features["MinTemp"]>-2.3)]
    features = features[(features["MaxTemp"]<2.3)&(features["MaxTemp"]>-2)]
    features = features[(features["Rainfall"]<4.5)]
    features = features[(features["Evaporation"]<2.8)]
    features = features[(features["Sunshine"]<2.1)]
    features = features[(features["WindGustSpeed"]<4)&(features["WindGustSpeed"]>-4)]
    features = features[(features["WindSpeed9am"]<4)]
    features = features[(features["WindSpeed3pm"]<2.5)]
    features = features[(features["Humidity9am"]>-3)]
    features = features[(features["Humidity3pm"]>-2.2)]
    features = features[(features["Pressure9am"]< 2)&(features["Pressure9am"]>-2.7)]
    features = features[(features["Pressure3pm"]< 2)&(features["Pressure3pm"]>-2.7)]
    features = features[(features["Cloud9am"]<1.8)]
    features = features[(features["Cloud3pm"]<2)]
    features = features[(features["Temp9am"]<2.3)&(features["Temp9am"]>-2)]
    features = features[(features["Temp3pm"]<2.3)&(features["Temp3pm"]>-2)]

    logger.debug("Splitting data")


    # Notebook code
    X = features.drop(["RainTomorrow"], axis=1)
    y = features["RainTomorrow"]

    # Splitting test and training sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)
    X_dev, X_test, y_dev, y_test = train_test_split(X_test, y_test, test_size = 0.5)

    # concatenate fo caching
    train = pd.concat([y_train, X_train], axis=1)
    validation = pd.concat([y_dev, X_dev], axis=1)
    test = pd.concat([y_test, X_test], axis=1)

    # SM template code
    # logger.info("Applying transforms.")
    # y = df.pop("rings")
    # X_pre = preprocess.fit_transform(df)
    # y_pre = y.to_numpy().reshape(len(y), 1)

    # X = np.concatenate((y_pre, X_pre), axis=1)
    # X = features.to_numpy()

    # logger.info("Splitting %d rows of data into train, validation, test datasets.", len(X))
    # np.random.shuffle(features)
    # train, validation, test = np.split(X, [int(0.7 * len(X)), int(0.85 * len(X))])

    logger.info("Writing out datasets to %s.", base_dir)
    train.to_csv(f"{base_dir}/train/train.csv", header=True, index=False)
    validation.to_csv(
        f"{base_dir}/validation/validation.csv", header=True, index=False
    )
    test.to_csv(f"{base_dir}/test/test.csv", header=True, index=False)
    pathlib.Path(f"{base_dir}/preprocess").mkdir(parents=True, exist_ok=True)
    joblib.dump(s_scaler, "model.joblib")
    with tarfile.open(f"{base_dir}/preprocess/model.tar.gz", "w:gz") as tf:
        tf.add(f"model.joblib")


def input_fn(input_data, content_type):
    """Parse input data payload

    We currently only take csv input. Since we need to process both labelled
    and unlabelled data we first determine whether the label column is present
    by looking at how many columns were provided.
    """
    if content_type == "application/json":
        return json.loads(input_data)["data"]
    else:
        raise ValueError("{} not supported by script!".format(content_type))


def output_fn(prediction, accept):
    """Format prediction output

    The default accept/content-type between containers for serial inference is JSON.
    We also want to set the ContentType or mimetype as the same value as accept so the next
    container can read the response payload correctly.
    """
    if accept == "application/json":
        instances = []
        for row in prediction.tolist():
            instances.append(row)
        json_output = {"data": instances}

        return json.dumps(json_output)
    else:
        raise RuntimeError("{} accept type is not supported by this script.".format(accept))


def predict_fn(input_data, model):
    """Preprocess input data

    We implement this because the default predict_fn uses .predict(), but our model is a preprocessor
    so we want to use .transform().

    The output is returned in the following order:

        rest of features either one hot encoded or standardized
    """
    features = model.transform(input_data)

    return features


def model_fn(model_dir):
    """Deserialize fitted model"""
    preprocessor = joblib.load(os.path.join(model_dir, "model.joblib"))
    return preprocessor