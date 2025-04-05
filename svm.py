import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler
from again import readcsv, create_sequences  # Assuming you have this from the previous script

SEQ_LENGTH = 10

def main():
    # Load & preprocess data
    df = readcsv("./Car_Hacking_Challenge_Dataset_rev20Mar2021/0_Preliminary/0_Training/Pre_train_D_1.csv") 
    X, y = create_sequences(df, seq_length=SEQ_LENGTH)

    # Flatten the time series (SVM expects 2D)
    X_flat = X.reshape(X.shape[0], -1)

    # Normalize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_flat)

    # Split into train/test (e.g., 80/20)
    split_idx = int(0.8 * len(X_scaled))
    X_train, X_test = X_scaled[:split_idx], X_scaled[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    # Train SVM
    clf = SVC(kernel='rbf', probability=True)
    clf.fit(X_train, y_train)

    # Predict
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]

    # Metrics
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("ROC AUC:", roc_auc_score(y_test, y_prob))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    print("ROC AUC Score:", roc_auc_score(y_test, y_prob))
    print(" Precision:", precision_score(y_test, y_pred))
    print(" Recall:", recall_score(y_test, y_pred))
    print(" F1 Score:", f1_score(y_test, y_pred))

if __name__ == "__main__":
    main()
