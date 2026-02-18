import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import flwr as fl
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from typing import Dict
from datetime import datetime
import seaborn as sns
import matplotlib.pyplot as plt
import threading


# 1. Initial Setup

NUM_CLIENTS = 10
NUM_ROUNDS = 5
DATASET_PATH = r'D:\CSE-CIC-IDS2018-parquet'
RESULTS_FILE = 'federated_model_results.txt'
PLOT_FILENAME = 'confusion_matrix_federated.png'
SERVER_ADDRESS = "127.0.0.1:8080"


# (Data Loading, Model Definition, and FlowerClient classes remain the same)
def load_and_preprocess_data(path):
    print("Step 1: Loading and Preprocessing Data...")
    all_files = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.parquet')]
    df = pd.concat([pd.read_parquet(file) for file in all_files], ignore_index=True)
    df['Label'] = df['Label'].apply(lambda x: 0 if x == 'Benign' else 1)
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].astype('category').cat.codes
    for col in df.columns:
        if df[col].dtype == 'float64':
            df[col] = df[col].astype('float32')
    X = df.drop(columns=['Label'])
    y = df['Label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    print("Step 2: Partitioning data across clients...")
    train_partitions = []
    for i in range(NUM_CLIENTS):
        client_slice = np.arange(len(X_train))[i::NUM_CLIENTS]
        x_part = X_train[client_slice]
        y_part = y_train.iloc[client_slice]
        train_partitions.append((x_part, y_part))
    test_set = (X_test, y_test)
    print("Data loading, preprocessing, and partitioning complete.")
    return train_partitions, test_set


def create_optimized_model(input_dim):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(input_dim,)),
        tf.keras.layers.Dense(units=138, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(units=106, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(units=32, activation='relu'),
        tf.keras.layers.Dense(units=1, activation='sigmoid')
    ])
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return model


class FlowerClient(fl.client.NumPyClient):
    def __init__(self, model, x_train, y_train):
        self.model = model
        self.x_train, self.y_train = x_train, y_train

    def get_parameters(self, config):
        return self.model.get_weights()

    def fit(self, parameters, config):
        self.model.set_weights(parameters)
        self.model.fit(self.x_train, self.y_train, epochs=1, batch_size=1024, verbose=0)
        return self.model.get_weights(), len(self.x_train), {}

    def evaluate(self, parameters, config):
        return 0.0, 0, {}


def get_evaluate_fn(model, x_test, y_test):
    def evaluate(server_round: int, parameters: fl.common.NDArrays, config: Dict[str, fl.common.Scalar]):
        model.set_weights(parameters)
        loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
        print(f"Round {server_round}: Centralized evaluation accuracy = {accuracy:.4f}")
        if server_round == NUM_ROUNDS:
            print("\nFinal round reached. Evaluating final global model...")
            y_pred_proba = model.predict(x_test)
            y_pred = (y_pred_proba > 0.5).astype(int)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            print(f"--- Final Federated Model Evaluation Results ---")
            print(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}")
            save_federated_results(RESULTS_FILE, accuracy, precision, recall, f1)
            cm = confusion_matrix(y_test, y_pred)
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Benign', 'Attack'],
                        yticklabels=['Benign', 'Attack'])
            plt.xlabel('Predicted Label')
            plt.ylabel('True Label')
            plt.title(f'Confusion Matrix - Federated Model (FedAvg, {NUM_ROUNDS} Rounds)')
            plt.savefig(PLOT_FILENAME)
            print(f"Final plot has been saved to '{PLOT_FILENAME}'")
        return loss, {"accuracy": accuracy}

    return evaluate


def save_federated_results(filename, accuracy, precision, recall, f1):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(filename, 'w') as f:
        f.write("--- Federated Model (FedAvg) Final Evaluation ---\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Number of Clients: {NUM_CLIENTS}\n")
        f.write(f"Number of Rounds: {NUM_ROUNDS}\n\n")
        f.write("Evaluation on Centralized Test Set:\n")
        f.write(f"- Accuracy:  {accuracy:.4f}\n")
        f.write(f"- Precision: {precision:.4f}\n")
        f.write(f"- Recall:    {recall:.4f}\n")
        f.write(f"- F1-Score:  {f1:.4f}\n")
        f.write("--------------------------------------------------\n")
    print(f"Final numerical results have been saved to '{filename}'")



# Main Execution Block

if __name__ == "__main__":
    # 1. Load data and create a model for the server's evaluation function
    train_partitions, test_set = load_and_preprocess_data(DATASET_PATH)
    X_test, y_test = test_set
    input_dim = train_partitions[0][0].shape[1]
    server_eval_model = create_optimized_model(input_dim)

    # 2. Define the strategy
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,
        min_available_clients=NUM_CLIENTS,
        evaluate_fn=get_evaluate_fn(server_eval_model, X_test, y_test),

        fraction_evaluate=0.0,
        min_evaluate_clients=0,
    )


    # 3. Function to start a single client in a background thread
    def run_client(cid: int):
        print(f"Starting client {cid} in a background thread...")
        x_train_client, y_train_client = train_partitions[cid]
        client_model = create_optimized_model(input_dim)
        client = FlowerClient(client_model, x_train_client, y_train_client)
        fl.client.start_numpy_client(server_address=SERVER_ADDRESS, client=client)
        print(f"Client {cid} has finished.")


    # 4. Start all clients in background threads
    client_threads = []
    for i in range(NUM_CLIENTS):
        thread = threading.Thread(target=run_client, args=(i,))
        client_threads.append(thread)
        thread.start()

    # 5. Start the server in the main thread (this is a blocking call)
    print("\nStarting Flower server in the main thread...")
    fl.server.start_server(
        server_address=SERVER_ADDRESS,
        config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
        strategy=strategy,
    )

    print("\nServer has finished. Waiting for all client threads to complete...")
    for thread in client_threads:
        thread.join()  # Wait for each client thread to finish

    print("Federated learning process has completed.")
    # The plot is saved by the evaluate_fn, but we can show it here if needed.
    # plt.show()
