import os
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import tensorflow as tf
from scikeras.wrappers import KerasClassifier
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import loguniform, randint


# 1. Initial Setup

DATASET_PATH = r'D:\CSE-CIC-IDS2018-parquet'
FINAL_RESULTS_FILE = 'optimized_model_results.txt'
PLOT_FILENAME = 'confusion_matrix_optimized.png'
SEARCH_LOG_FILE = 'hyperparameter_search_log.csv'



# 2. Data Loading and Preprocessing

def load_and_combine_data(path):
    print(f"Reading data from {path}...")
    all_files = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.parquet')]
    if not all_files:
        raise FileNotFoundError(f"No .parquet files found in path '{path}'.")
    df_list = [pd.read_parquet(file) for file in all_files]
    full_df = pd.concat(df_list, ignore_index=True)
    print("All files successfully combined.")
    return full_df


def preprocess_data(df):
    print("Starting data preprocessing...")
    df['Label'] = df['Label'].apply(lambda x: 0 if x == 'Benign' else 1)
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].astype('category').cat.codes
    print("Downcasting float64 columns to float32 to save memory...")
    for col in df.columns:
        if df[col].dtype == 'float64':
            df[col] = df[col].astype('float32')
    print("Preprocessing finished.")
    return df



# 3. Model Building Function (Corrected Version)

def create_mlp_model(input_dim, neurons_l1=128, neurons_l2=64, dropout_rate=0.2, learning_rate=0.001):
    """Creates a Keras MLP model with a defined input shape."""
    model = tf.keras.models.Sequential([
        # Explicitly define the input shape to prevent the AttributeError
        tf.keras.layers.Input(shape=(input_dim,)),
        tf.keras.layers.Dense(neurons_l1, activation='relu'),
        tf.keras.layers.Dropout(dropout_rate),
        tf.keras.layers.Dense(neurons_l2, activation='relu'),
        tf.keras.layers.Dropout(dropout_rate),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return model



def save_final_results(filename, best_params, accuracy, precision, recall, f1):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(filename, 'w') as f:
        f.write("--- Optimized Model (Centralized) Evaluation Results ---\n")
        f.write(f"Timestamp: {timestamp}\n\n")
        f.write("Best Hyperparameters Found:\n")
        for key, value in best_params.items():
            f.write(f"- {key}: {value}\n")
        f.write("\nEvaluation on Unseen Test Set:\n")
        f.write(f"- Accuracy:  {accuracy:.4f}\n")
        f.write(f"- Precision: {precision:.4f}\n")
        f.write(f"- Recall:    {recall:.4f}\n")
        f.write(f"- F1-Score:  {f1:.4f}\n")
        f.write("----------------------------------------------------------\n")
    print(f"\nFinal results have been saved to '{filename}'")




# 5. Main function for the optimization process

def main():
    df = load_and_combine_data(DATASET_PATH)
    df = preprocess_data(df)

    X = df.drop(columns=['Label'])
    y = df['Label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Get the input dimension from the training data
    input_dim = X_train.shape[1]

    print("\nSetting up hyperparameter search space...")
    # Pass the input_dim to the KerasClassifier
    model_wrapper = KerasClassifier(
        model=create_mlp_model,
        input_dim=input_dim,
        verbose=0,
        loss='binary_crossentropy'
    )

    param_dist = {
        'model__neurons_l1': randint(64, 257),
        'model__neurons_l2': randint(32, 129),
        'model__dropout_rate': [0.2, 0.3, 0.4, 0.5],
        'model__learning_rate': loguniform(1e-4, 1e-2),
        'batch_size': [1024, 2048, 4096]
    }


    search = RandomizedSearchCV(
        estimator=model_wrapper,
        param_distributions=param_dist,
        n_iter=10,
        cv=3,
        scoring='f1',
        verbose=3,
        random_state=42,
        n_jobs=4
    )

    print("\nStarting Hyperparameter Search (Quick Test - Attempt 2)...")
    search.fit(X_train, y_train, epochs=10)

    print("\nSearch finished!")

    print(f"Best F1-score found during search: {search.best_score_:.4f}")
    print("Best parameters found:")
    print(search.best_params_)

    print(f"\nSaving full hyperparameter search log to '{SEARCH_LOG_FILE}'...")
    search_results_df = pd.DataFrame(search.cv_results_)
    search_results_df.sort_values(by='rank_test_score', inplace=True)
    search_results_df.to_csv(SEARCH_LOG_FILE, index=False)

    print("\nEvaluating the best model on the unseen test set...")
    best_model = search.best_estimator_
    y_pred = best_model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print("\n--- Optimized Model Final Evaluation Results ---")
    print(f"Accuracy:  {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}")
    save_final_results(FINAL_RESULTS_FILE, search.best_params_, accuracy, precision, recall, f1)

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Benign', 'Attack'],
                yticklabels=['Benign', 'Attack'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix - Optimized Centralized Model')

    plt.savefig(PLOT_FILENAME)
    print(f"Plot has been saved to '{PLOT_FILENAME}'")
    plt.show()


if __name__ == '__main__':
    main()
