import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve,
    precision_score,
    recall_score,
    f1_score
)
from tensorflow.keras.models import load_model

# Import shared functions
from again import readcsv, create_sequences, SEQ_LENGTH


def plot_roc_curve(y_true, y_scores):
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc_score(y_true, y_scores):.2f}")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.grid()
    plt.show()

def plot_conf_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure()
    plt.imshow(cm, cmap='Blues')
    plt.title("Confusion Matrix")
    plt.colorbar()
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    for i in range(2):
        for j in range(2):
            plt.text(j, i, cm[i, j], ha='center', va='center', color='red')
    plt.show()


def main():
    model = load_model("models/2epochmanual_federated_lstm_model.h5")
    df_test = readcsv("./Car_Hacking_Challenge_Dataset_rev20Mar2021/0_Preliminary/0_Training/Pre_train_D_2.csv")
    X_test, y_test = create_sequences(df_test, SEQ_LENGTH)
    print("predicting values")
    y_probs = model.predict(X_test).flatten()
    y_pred = (y_probs > 0.5).astype(int)

    print("ROC AUC:", roc_auc_score(y_test, y_probs))
    print("Precision:", precision_score(y_test, y_pred))
    print("Recall:", recall_score(y_test, y_pred))
    print("F1 Score:", f1_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    plot_roc_curve(y_test, y_probs)
    plot_conf_matrix(y_test, y_pred)

if __name__ == "__main__":
    main()