EPOCHS = 50



SEQUENCE_LENGTH = 30

LEARNING_RATES = [0.001, 0.0005]
MODEL_NAMES = ["LSTM_Model_01", "LSTM_Model_02"]

OP_COLS = ["op_setting_1", "op_setting_2", "op_setting_3"]
SENSOR_COLS = [f"sensor_measurement_{i}" for i in range(1, 22)]
