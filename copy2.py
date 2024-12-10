import os
import psutil
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Function to measure the execution time of an algorithm
def measure_time(func, *args):
    start = time.time()
    func(*args)
    end = time.time()
    return end - start

# Experiment 1: Matrix Multiplication
def matrix_multiplication():
    A = np.random.rand(500, 500)
    B = np.random.rand(500, 500)
    C = np.dot(A, B)

# Experiment 2: Random Forest Classifier
def random_forest_classifier(X_train, y_train, X_test, y_test):
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)
    return accuracy

# Function to perform the experiment with different core limits
def perform_experiment(cpu_core_limits):
    matrix_times = []
    rf_times = []
    rf_accuracies = []

    # Create a sample dataset for Random Forest
    X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Run the experiments for each core limit
    for cores in cpu_core_limits:
        print(f"\nExperimenting with {cores} CPU cores...")
        
        # Limit CPU cores for the current experiment
        os.environ["OMP_NUM_THREADS"] = str(cores)

        # Measure execution time for Matrix Multiplication
        matrix_time = measure_time(matrix_multiplication)
        matrix_times.append(matrix_time)
        print(f"Matrix Multiplication Execution Time with {cores} cores: {matrix_time:.4f} seconds")

        # Measure execution time and accuracy for Random Forest
        rf_accuracy = random_forest_classifier(X_train, y_train, X_test, y_test)
        rf_times.append(matrix_time)
        rf_accuracies.append(rf_accuracy)
        print(f"Random Forest Accuracy with {cores} cores: {rf_accuracy:.4f}")

    return matrix_times, rf_accuracies

# List of CPU core configurations (1, 2, 4, 8 cores)
cpu_core_limits = [1, 2, 4, 8]

# Running the experiment
matrix_times, rf_accuracies = perform_experiment(cpu_core_limits)

# Plotting and Saving Results for Matrix Multiplication
plt.figure(figsize=(10, 6))
plt.plot(cpu_core_limits, matrix_times, marker='o', linestyle='-', color='b', label="Matrix Multiplication Time")
plt.xlabel("Number of CPU Cores")
plt.ylabel("Execution Time (seconds)")
plt.title("Effect of CPU Cores on Matrix Multiplication Performance")
plt.grid(True)
plt.legend()
plt.savefig("matrix_multiplication_performance.png")  # Save the figure
plt.show()
plt.close()

# Plotting and Saving Results for Random Forest Classifier Accuracy
plt.figure(figsize=(10, 6))
plt.plot(cpu_core_limits, rf_accuracies, marker='s', linestyle='-', color='g', label="Random Forest Accuracy")
plt.xlabel("Number of CPU Cores")
plt.ylabel("Accuracy")
plt.title("Effect of CPU Cores on Random Forest Performance")
plt.grid(True)
plt.legend()
plt.savefig("random_forest_performance.png")  # Save the figure
plt.show()
plt.close()

# Example: Output available CPU information
print(f"Total number of available CPU cores: {psutil.cpu_count()}")
