import os
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint , EarlyStopping
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K
import matplotlib.pyplot as plt




DATA_PATH = "CMaps/train_FD001.txt"
OP_COLS = ["op1", "op2", "op3"]
SENSOR_COLS = [f"sensor{i}" for i in range(1, 22)]  # sensors 1 to 21
ALL_COLUMNS = ["unit", "time_cycles"] + OP_COLS + SENSOR_COLS


SEQUENCE_LENGTH = 25




def get_callbacks(model_name):
    tensorboard_cb = TensorBoard(log_dir=f"logs/{model_name}")
    checkpoint_cb = ModelCheckpoint(f"saved_models/{model_name}.h5", save_best_only=True)
    return [tensorboard_cb, checkpoint_cb]



def asymmetric_loss(y_true, y_pred):
    error = y_pred - y_true
    mask = K.cast(K.less(error, 0.0), K.floatx())
    loss_early = K.exp(-error / 13.0) - 1
    loss_rate = K.exp(error / 10) - 1
    return K.mean(mask * loss_early + (1 - mask) * loss_rate)





def load_and_preprocess_data(filepath):
    df = pd.read_csv(filepath, sep="\s+", header=None, names=ALL_COLUMNS)

    # Compute RUL
    df["max_cycle"] = df.groupby("unit")["time_cycles"].transform("max")
    df["RUL"] = df["max_cycle"] - df["time_cycles"]
    df.drop("max_cycle", axis=1, inplace=True)

    return df
def normalize(df, feature_columns):
    scaler = MinMaxScaler()
    df[feature_columns] = scaler.fit_transform(df[feature_columns])
    return df, scaler


def create_sequences(df, sequence_length, feature_columns):
    sequences = []
    rul_values = []

    for unit in df["unit"].unique():
        unit_df = df[df["unit"] == unit]
        unit_df = unit_df.reset_index(drop=True)
        for i in range(len(unit_df) - sequence_length + 1):
            seq = unit_df.loc[i : i + sequence_length - 1, feature_columns].values
            label = unit_df.loc[i + sequence_length - 1, "RUL"]
            sequences.append(seq)
            rul_values.append(label)

    X = np.array(sequences)
    y = np.array(rul_values).reshape(-1, 1)

    return X, y

def plot_loss(history):
    plt.plot(history.history["loss"], label="Train Loss")
    plt.plot(history.history["val_loss"], label="Val Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.legend()
    plt.grid()
    plt.show()




def get_callbacks(model_name):
    return [
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        ModelCheckpoint(f"models/{model_name}.h5", save_best_only=True)
    ]




OP_SETTING_COLUMNS = ['op_setting_{}'.format(x) for x in range(1, 4)]
SENSOR_COLUMNS = ['sensor_{}'.format(x) for x in range(1, 22)]

CURRENT_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = os.path.join(CURRENT_DIR, 'CMaps')


def read_data(filepath):
    '''
    Reads `filepath` as space separated file and returns pd.DataFrame
    '''
    col_names = ['unit', 'time_cycles'] + OP_SETTING_COLUMNS + SENSOR_COLUMNS
    return pd.read_csv(
        filepath,
        sep='\s+',
        header=None,
        names=col_names
    )

def read_dataset(dataset_name):
    '''
    Reads TRAIN, TEST and RUL datasets for specified dataset name

    Parameters
    ----------
    dataset_name : str, name of the dataset, e.g. 'FD001'

    Returns
    -------
    a tuple of (pd.DataFrame, pd.DataFrame, np.array) for TRAIN, TEST AND RUL
    datasets correspondingly
    '''
    TRAIN_FILE = os.path.join(DATA_DIR, f'train_{dataset_name}.txt')
    TEST_FILE = os.path.join(DATA_DIR, f'test_{dataset_name}.txt')
    TEST_RUL_FILE = os.path.join(DATA_DIR, f'RUL_{dataset_name}.txt')

    train_data = read_data(TRAIN_FILE)
    test_data = read_data(TEST_FILE)
    test_rul = np.loadtxt(TEST_RUL_FILE)

    return train_data, test_data, test_rul


def calculate_RUL(X, upper_threshold=None):
    '''
    Calculate Remaining Useful Life per `unit`

    Parameters
    ----------
    X : pd.DataFrame, with `unit` and `time_cycles` columns
    upper_threshold: int, limit maximum RUL valus, default is None

    Returns
    -------
    np.array with Remaining Useful Life values
    '''
    lifetime = X.groupby(['unit'])['time_cycles'].transform(max)
    rul = lifetime - X['time_cycles']

    if upper_threshold:
        rul = np.where(rul > upper_threshold, upper_threshold, rul)

    return rul