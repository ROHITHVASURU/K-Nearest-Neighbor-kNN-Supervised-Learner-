import numpy as np
from math import sqrt
from collections import Counter
from sklearn.model_selection import KFold
import csv

# Load the labeled dataset (exclude the name field for distance calculations)
def load_data(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split()
            cls = int(parts[0])
            x = float(parts[1])
            y = float(parts[2])
            data.append((cls, x, y))
    return data

'''# Data loading function for the fertility dataset
def load_data(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split(',')  # Split by comma
            # Convert all features to floats, except the last column (class label)
            features = [float(value) for value in parts[:-1]]
            # Convert the class label ('N' -> 0, 'O' -> 1)
            cls = 0 if parts[-1] == 'N' else 1
            data.append((cls, *features))  # Store class and features as a tuple
    return data
'''

# Euclidean distance function
def euclidean_distance(point1, point2):
    return sqrt(sum((p1 - p2) ** 2 for p1, p2 in zip(point1[1:], point2[1:])))

# kNN classification function (Store All variant)
def knn_classify(instance, training_data, k=3):
    distances = [(train_instance, euclidean_distance(instance, train_instance)) for train_instance in training_data]
    distances.sort(key=lambda x: x[1])  # Sort by distance
    k_nearest = distances[:k]  # Get k nearest neighbors
    k_nearest_classes = [instance[0] for instance, _ in k_nearest]
    return Counter(k_nearest_classes).most_common(1)[0][0]  # Return majority class

# Evaluate accuracy for a dataset
def evaluate_accuracy(training_data, test_data, k=3):
    correct = 0
    for instance in test_data:
        predicted_class = knn_classify(instance, training_data, k)
        if predicted_class == instance[0]:
            correct += 1
    return correct / len(test_data)

# Cross-validation function (5-fold)
def cross_validation(data, k=3, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True)
    accuracies_train = []
    accuracies_test = []

    for train_index, test_index in kf.split(data):
        training_data = [data[i] for i in train_index]
        test_data = [data[i] for i in test_index]

        # Calculate accuracy for train and test data
        train_acc = evaluate_accuracy(training_data, training_data, k)
        test_acc = evaluate_accuracy(training_data, test_data, k)

        accuracies_train.append(train_acc)
        accuracies_test.append(test_acc)

    return np.mean(accuracies_train), np.mean(accuracies_test)

# "Store Errors" kNN variant
def store_errors_knn(training_data, k=3):
    stored_instances = []
    for instance in training_data:
        if len(stored_instances) < k:
            stored_instances.append(instance)
        else:
            predicted_class = knn_classify(instance, stored_instances, k)
            if predicted_class != instance[0]:
                stored_instances.append(instance)
    return stored_instances

# Cross-validation for "Store Errors" variant
def cross_validation_store_errors(data, k=3, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True)
    accuracies_train = []
    accuracies_test = []

    for train_index, test_index in kf.split(data):
        training_data = [data[i] for i in train_index]
        test_data = [data[i] for i in test_index]

        # Train using "Store Errors" variant
        stored_data = store_errors_knn(training_data, k)

        # Calculate accuracy for train and test data
        train_acc = evaluate_accuracy(stored_data, training_data, k)
        test_acc = evaluate_accuracy(stored_data, test_data, k)

        accuracies_train.append(train_acc)
        accuracies_test.append(test_acc)

    return np.mean(accuracies_train), np.mean(accuracies_test)

# Main execution to test multiple k values
if __name__ == "__main__":
    # Load data from fertility dataset
    file_path = '/Users/rohith/Downloads/Project_1_Rohith/part-A/labeled-examples'  # Replace with the correct path
    data = load_data(file_path)

    # List of k values to test
    k_values = [3, 5, 7, 9]

    # Loop over different values of k and print results for both Store All and Store Errors variants
    for k_value in k_values:
        print(f"\nResults for k = {k_value}:")

        # Store All kNN with 5-fold cross-validation
        print(f"kNN (Store All variant) for k = {k_value}:")
        train_accuracy, test_accuracy = cross_validation(data, k=k_value, n_splits=5)
        print(f"Training Accuracy: {train_accuracy:.4f}, Testing Accuracy: {test_accuracy:.4f}")

        # Store Errors kNN with 5-fold cross-validation
        print(f"kNN (Store Errors variant) for k = {k_value}:")
        train_accuracy_store, test_accuracy_store = cross_validation_store_errors(data, k=k_value, n_splits=5)
        print(f"Training Accuracy: {train_accuracy_store:.4f}, Testing Accuracy: {test_accuracy_store:.4f}")
