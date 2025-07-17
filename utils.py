import os
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint

def get_callbacks(model_name):
    tensorboard_cb = TensorBoard(log_dir=f"logs/{model_name}")
    checkpoint_cb = ModelCheckpoint(f"saved_models/{model_name}.h5", save_best_only=True)
    return [tensorboard_cb, checkpoint_cb]
