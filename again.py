import numpy as np
import pandas as pd
import tensorflow as tf

SEQ_LENGTH = 20
NUM_CLIENTS = 3
NUM_ROUNDS = 3
FEATURE_DIM = 3
EPOCHS = 1
BATCH_SIZE = 32

def readcsv(filepath):
    df = pd.read_csv(filepath, comment='#', header=0, dtype={
        "Timestamp": float, "Arbitration_ID": str, "Data": str, "Class": str
    })

    df["Timestamp"] = df.groupby("Arbitration_ID")["Timestamp"].diff().fillna(0)
    df["Arbitration_ID"] = df["Arbitration_ID"].apply(lambda x: int(x, 16))
    df["Arbitration_ID"] = (
        (df["Arbitration_ID"] - df["Arbitration_ID"].min()) / 
        (df["Arbitration_ID"].max() - df["Arbitration_ID"].min())
    )

    def hex_to_float(hex_string):
        hex_string = hex_string.replace(" ", "")
        int_value = int(hex_string, 16)
        return int_value / (2**64 - 1)

    df["Data"] = df["Data"].apply(hex_to_float)
    df.drop(columns=["DLC"], inplace=True, errors='ignore')  # in case DLC isn't there
    df["Class"] = df["Class"].apply(lambda x: 0 if x == "Normal" else 1)

    print(f"Data loaded and preprocessed:\n{df.head(3)}")
    return df

def create_sequences(df, seq_length=10):
    feature_columns = ["Timestamp", "Arbitration_ID", "Data"]
    target_column = "Class"

    data = df[feature_columns].values
    target = df[target_column].values

    X = np.lib.stride_tricks.sliding_window_view(data, (seq_length, len(feature_columns)))[:, 0, :, :]
    y = target[seq_length - 1 : len(X) + seq_length - 1]

    return X, y

def create_lstm_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(SEQ_LENGTH, FEATURE_DIM)),
        tf.keras.layers.LSTM(16, return_sequences=True, activation="relu"),
        tf.keras.layers.LSTM(8, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid")
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

def average_weights(weights_list):
    return [np.mean(w, axis=0) for w in zip(*weights_list)]

def main():
    df = readcsv("./Car_Hacking_Challenge_Dataset_rev20Mar2021/0_Preliminary/0_Training/Pre_train_D_1.csv")
    X, y = create_sequences(df, SEQ_LENGTH)

    # Split data across clients
    client_X = np.array_split(X, NUM_CLIENTS)
    client_y = np.array_split(y, NUM_CLIENTS)

    global_model = create_lstm_model()

    for round_num in range(1, NUM_ROUNDS + 1):
        print(f"\nRound {round_num}")
        client_weights = []

        for i in range(NUM_CLIENTS):
            print(f" Client {i+1} training...")
            local_model = create_lstm_model()
            local_model.set_weights(global_model.get_weights())

            local_model.fit(client_X[i], client_y[i],
                            batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=1)
            client_weights.append(local_model.get_weights())

        # Average client weights
        global_weights = average_weights(client_weights)
        global_model.set_weights(global_weights)

        # Evaluate global model (optional)
        total_loss, total_acc, total_samples = 0.0, 0.0, 0
        for i in range(NUM_CLIENTS):
            loss, acc = global_model.evaluate(client_X[i], client_y[i], verbose=0)
            total_loss += loss * len(client_X[i])
            total_acc += acc * len(client_X[i])
            total_samples += len(client_X[i])
        print(f"Round {round_num} -- Loss: {total_loss / total_samples:.4f}, Accuracy: {total_acc / total_samples:.4f}")

    global_model.save("manual_federated_lstm_model.h5")
    print("Saved final model as 'manual_federated_lstm_model.h5'")

if __name__ == "__main__":
    main()
