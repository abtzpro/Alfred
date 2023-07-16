import pennylane as qml
from pennylane import numpy as np
from sklearn import datasets, preprocessing, model_selection
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import boto3
from botocore import UNSIGNED
from botocore.client import Config
import os

def create_quantum_model(params, x=None, n_qubits=8):
    """
    Function to create a quantum model.
    """
    qml.templates.AmplitudeEmbedding(features=x, wires=range(n_qubits), normalize=True)
    for i in range(len(params[0])):
        qml.templates.BasicEntanglerLayers(params[0][i], wires=range(n_qubits))
    qml.templates.StronglyEntanglingLayers(params[1], wires=range(n_qubits))

    return qml.qnode(dev)(create_quantum_model)

def create_quantum_features(params, X, n_qubits=8):
    """
    Function to create quantum features.
    """
    dev = qml.device("default.qubit", wires=n_qubits)
    qnode = create_quantum_model(params, X, n_qubits)
    
    return np.array([qnode(params, x) for x in X])

def download_cybersecurity_data(destination_folder='dest-dir', bucket_name='cse-cic-ids2018'):
    """
    Function to download cybersecurity dataset.
    """
    # Try creating a boto3 client.
    try:
        s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))
    except Exception as e:
        print(f"Failed to create a boto3 client due to: {e}")
        return
    
    # Try downloading the files.
    try:
        files_in_bucket = s3.list_objects(Bucket=bucket_name)['Contents']
        for file in files_in_bucket:
            file_name = file['Key']
            destination_file_path = os.path.join(destination_folder, file_name)
            s3.download_file(bucket_name, file_name, destination_file_path)
    except Exception as e:
        print(f"Failed to download data due to: {e}")
        return
# Part 2

def load_cybersecurity_data(destination_folder='dest-dir'):
    """
    Function to load and preprocess the cybersecurity dataset.
    """
    data_files = os.listdir(destination_folder)

    # Try reading each data file and concatenate them into a single DataFrame.
    try:
        data = pd.concat([pd.read_csv(os.path.join(destination_folder, file)) for file in data_files])
    except Exception as e:
        print(f"Failed to read data files due to: {e}")
        return

    # Handle missing values.
    data.fillna(data.median(), inplace=True)
    
    # Convert categorical variables to numerical.
    for col in data.columns[data.dtypes == 'object']:
        data[col] = LabelEncoder().fit_transform(data[col].astype(str))
    
    cyber_data = data.drop('Label', axis=1).values
    cyber_labels = data['Label'].values
    
    return cyber_data, cyber_labels

# Download the dataset
download_cybersecurity_data()

# Load the dataset
cyber_data, cyber_labels = load_cybersecurity_data()

# Split the data into training and testing sets.
X_train, X_test, y_train, y_test = model_selection.train_test_split(cyber_data, cyber_labels, test_size=0.2, random_state=42)

# Scale the data to the range [-pi, pi].
scaler = preprocessing.MinMaxScaler((-np.pi, np.pi))
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define a cost function for training the quantum model.
def cost_fn(params):
    X_train_quantum = create_quantum_features(params, X_train_scaled)
    model = SVC(kernel='linear')
    model.fit(X_train_quantum, y_train)
    accuracy = model.score(X_train_quantum, y_train)
    return 1 - accuracy

# Hyperparameter tuning for optimization
n_layers = 2
init_params = np.random.uniform(low=-np.pi, high=np.pi, size=(2, n_layers, n_qubits, 3))

grid_search = GridSearchCV(estimator=qml.GradientDescentOptimizer(),
                           param_grid={"learning_rate": [0.01, 0.1, 1]},
                           scoring="accuracy",
                           n_jobs=-1)
grid_search.fit(cost_fn, init_params)
opt = grid_search.best_estimator_

params = opt.step(cost_fn, init_params)

# Create quantum features for the optimized parameters.
X_train_quantum = create_quantum_features(params, X_train_scaled)
X_test_quantum = create_quantum_features(params, X_test_scaled)

# Train a classical SVC model and test it.
model = SVC(kernel='linear')
model.fit(X_train_quantum, y_train)
accuracy = model.score(X_test_quantum, y_test)

print('Test accuracy:', accuracy)
