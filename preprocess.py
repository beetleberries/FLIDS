import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import tensorflow as tf
import tensorflow.keras
from sklearn.metrics import classification_report
print(tensorflow.keras.__version__)
print(tf.__version__)
print(tf.keras.__version__) 
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint



def readcsv(filepath):
    df = pd.read_csv(filepath, comment='#', header=0, dtype={"Timestamp": float, "Arbitration_ID": str, "Data": str, "Class": str})
    


    df["Timestamp"] = df.groupby("Arbitration_ID")["Timestamp"].diff().fillna(0)

    df["Arbitration_ID"] = df["Arbitration_ID"].apply(lambda x: int(x, 16))
    df["Arbitration_ID"] = (df["Arbitration_ID"] - df["Arbitration_ID"].min()) / (df["Arbitration_ID"].max() - df["Arbitration_ID"].min())

    def hex_to_float(hex_string):
        hex_string = hex_string.replace(" ", "")
        int_value = int(hex_string, 16)
        return int_value / (2**64 - 1) 

    df["Data"] = df["Data"].apply(hex_to_float)

    df.drop(columns=["DLC"], inplace=True)
    df["Class"] = df["Class"].apply(lambda x: 0 if x == "Normal" else 1)

    print(f"data preprocessed and read from csv{df.head(10)}")
    return df 



def compile_model(SEQ_LENGTH, featurelength): 
    #force gpu
    physical_devices = tf.config.list_physical_devices("GPU")
    if physical_devices:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)

    # Build LSTM Model
    model = Sequential([
        LSTM(64, return_sequences=True, activation="relu", input_shape=(SEQ_LENGTH, featurelength)),
        Dropout(0.2),
        LSTM(64, return_sequences=False, activation="relu"),
        Dropout(0.5),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')  # Binary classification
    ])

    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

    return model
    

def create_sequences(data, target_column, seq_length=10):
    X = np.lib.stride_tricks.sliding_window_view(data.values, (seq_length, data.shape[1]))[:, 0, :, :]
    y = target_column.iloc[seq_length-1:len(X) + seq_length-1].values
    return X, y

def main():
    df = readcsv("./Car_Hacking_Challenge_Dataset_rev20Mar2021/0_Preliminary/0_Training/Pre_train_D_1.csv")
    
    print(f"The amount of anomalies is {df['Class'].sum() / len(df) * 100:.2f}% currently")

    SEQ_LENGTH = 50
    features = ["Timestamp", "Arbitration_ID", "Data"]

    print(df.columns)
    X, y = create_sequences(df[features], df["Class"], seq_length=SEQ_LENGTH)


    print("splitting")
    # Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

    # Reshape for LSTM (samples, timesteps, features)
    X_train = np.reshape(X_train, (X_train.shape[0], SEQ_LENGTH, len(features)))
    X_test = np.reshape(X_test, (X_test.shape[0], SEQ_LENGTH, len(features)))

    model = compile_model(SEQ_LENGTH, len(features))



    print("training")
    early_stop = EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True)
    checkpoint = ModelCheckpoint(
    "models/lstm_most_recent_best_model.weights.h5", monitor="val_loss", save_best_only=True, save_weights_only=True)
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=50, callbacks=[early_stop, checkpoint])

    # Evaluate the model
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"Test Accuracy: {accuracy:.4f}")

    model.save("models/lstm_can_model.h5")

    print(classification_report(y_test, model.predict(X_test).round()))
    return

if __name__ == "__main__":
    main()