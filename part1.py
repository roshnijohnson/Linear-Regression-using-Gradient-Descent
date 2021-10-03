# Imported libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


# reading dataset from UCI

dataset = pd.read_excel(
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00477/Real%20estate%20valuation%20data%20set.xlsx")
print(dataset.head())

# Checking dataset

print(dataset.info())

#  Removing the 'No' column
dataset = dataset.drop('No', 1)

# checking for null values
dataset.isna().sum()

# Converting feature 'Transaction date' to type int

dataset['X1 transaction date'] = dataset['X1 transaction date'].astype(int)

# Calculating the Correlation

dataset.corr()["Y house price of unit area"][:-1:].sort_values(ascending=False)

# Plotting correlation graph

plt.figure(figsize=(10, 8))
sns.heatmap(data=dataset.corr(), annot=True, cmap="viridis")
plt.show()

# Splitting X (attributes) and Y (Output)

X = dataset.iloc[:, :-1]
Y = dataset.iloc[:, -1:]

# Splitting Train and Test data

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.20, random_state=0)

# Normalizing the X_train and X_test

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)

# Funciton to calculate partial derivate


def calc_partialderivate(total_samples, X, y_true, y_pred):
    w_grad = -(2/total_samples) * (X.T.dot(y_true-y_pred))
    b_grad = -(2/total_samples) * np.sum(y_true-y_pred)

    return w_grad, b_grad

# Function to calculate gradient descent


def gradient_descent(X, y_true, epochs, learning_rate):

    features = X.shape[1]           # No. of features

    w = np.ones(shape=(features))   # Initializing the weights with 1
    b = 0                           # Initializing the bais with 0
    total_samples = X.shape[0]      # Count of total number of samples

    # To store epoch,cost,R2,MSE for different iteration and parameters
    list_cost = []
    list_epoch = []
    r2_list = []
    r2 = []

    mse_list = []
    mse = []

    cost = 0

    for i in range(epochs):

        # calculating w1X1 + w2X2 + ... + Bias

        y_pred = np.dot(w, X.T) + b

        #  calling partial derivative function of w and b

        w_grad, b_grad = calc_partialderivate(total_samples, X, y_true, y_pred)

        # updating the w and b

        w = w - (learning_rate * w_grad)
        b = b - (learning_rate * b_grad)

        # Calculating cost

        cost = np.mean(np.square(y_true-y_pred))

        if i % 100 == 0:
            list_cost.append(cost)
            list_epoch.append(i)
        r2_list.append(r2_score(y_true, y_pred) * 100)
        mse_list.append(mean_squared_error(y_true, y_pred))

    r2.append(np.mean(r2_list))
    mse.append(np.mean(mse_list))

    return w, b, cost, r2, mse


# To store the results for each iteration
w_list = {}
b_list = {}
cost_values = {}
r2_sc = {}
mse_sc = {}

# Initializing the learning_rate and epochs
learning_rate = np.arange(0.01, 0.1, 0.01)
epochs = np.arange(4000, 10000, 1000)

# Calling gradient_descent for multiple epochs and learning_rates on Training data
for learn_rate in learning_rate:
    for epoch in epochs:
        w, b, cost, r22, mse22 = gradient_descent(X_train, np.array(
            Y_train).reshape(Y_train.shape[0],), epoch, learn_rate)
        w_list[(learn_rate, epoch)] = w
        b_list[(learn_rate, epoch)] = b
        cost_values[(learn_rate, epoch)] = cost
        r2_sc[(learn_rate, epoch)] = r22
        mse_sc[(learn_rate, epoch)] = mse22

print("----------Working on Training dataset----------")
print("List of Coefficients for each iteration: ", w_list)
print("List of Bias for each iteration: ", b_list)
print("List of Cost for each iteration: ", cost_values)
print("List of R2_Score for each iteration: ", r2_sc)
print("List of MSE for each iteration: ", mse_sc)


# Calling gradient_descent for multiple epochs and learning_rates on testing data

test_w_list = {}
test_b_list = {}
test_cost_values = {}
test_r2_sc = {}
test_mse_sc = {}

learning_rate = np.arange(0.01, 0.1, 0.01)
epochs = np.arange(4000, 10000, 1000)

for learn_rate in learning_rate:
    for epoch in epochs:
        w, b, cost, r22, mse22 = gradient_descent(X_test, np.array(
            Y_test).reshape(Y_test.shape[0],), epoch, learn_rate)
        test_w_list[(learn_rate, epoch)] = w
        test_b_list[(learn_rate, epoch)] = b
        test_cost_values[(learn_rate, epoch)] = cost
        test_r2_sc[(learn_rate, epoch)] = r22
        test_mse_sc[(learn_rate, epoch)] = mse22

print("----------Working on Testing dataset----------")
print("List of Coefficients for each iteration: ", test_w_list)
print("List of Bias for each iteration: ", test_b_list)
print("List of Cost for each iteration: ", test_cost_values)
print("List of R2_Score for each iteration: ", test_r2_sc)
print("List of MSE for each iteration: ", test_mse_sc)
# Plotting graphs for Training data

# Graph 1
sns.set_style("darkgrid")
plt.figure(figsize=(18, 10))

plt.plot(np.arange(len(learning_rate) * len(epochs)),
         list(cost_values.values()), color="red")
plt.xticks(np.arange(len(learning_rate) * len(epochs)),
           labels=cost_values.keys(), rotation='vertical')

plt.title("Training dataset Learning Rate - Epoch Vs Cost")

plt.xlabel("Learning Rate - Epoch")
plt.ylabel("Cost")
plt.tight_layout()
plt.draw()
plt.show()

# Graph 2
sns.set_style("darkgrid")
plt.figure(figsize=(18, 10))

plt.plot(np.arange(len(learning_rate) * len(epochs)),
         list(r2_sc.values()), color="red")
plt.xticks(np.arange(len(learning_rate) * len(epochs)),
           labels=r2_sc.keys(), rotation='vertical')

plt.title("Training dataset Learning Rate - Epoch Vs R2_Score")

plt.xlabel("Learning Rate - Epoch")
plt.ylabel("R2_Score")

plt.tight_layout()
plt.draw()
plt.show()

# Graph 3

sns.set_style("darkgrid")
plt.figure(figsize=(18, 10))

plt.plot(np.arange(len(learning_rate) * len(epochs)),
         list(mse_sc.values()), color="red")
plt.xticks(np.arange(len(learning_rate) * len(epochs)),
           labels=mse_sc.keys(), rotation='vertical')

plt.title("Training dataset Learning Rate - Epoch Vs MSE")

plt.xlabel("Learning Rate - Epoch")
plt.ylabel("MSE")

plt.tight_layout()
plt.draw()
plt.show()

# Calculating Best value for Cost, R2_Score and MSE for training dataset
print("----------Training dataset----------")

print("Minimum Cost Value on Training Data: ", min(cost_values.values()))
print("Maximum R2-Score Value on Training Data: ", max(r2_sc.values()))
print("Minimum MSE Value on Training Data: ", min(mse_sc.values()))

# Plotting graphs for Testing data

# Graph 1
sns.set_style("darkgrid")
plt.figure(figsize=(18, 10))

plt.plot(np.arange(len(learning_rate) * len(epochs)),
         list(test_cost_values.values()), color="blue")
plt.xticks(np.arange(len(learning_rate) * len(epochs)),
           labels=test_cost_values.keys(), rotation='vertical')

plt.title("Testing dataset Learning Rate - Epoch Vs Cost")
plt.xlabel("Learning Rate - Epoch")
plt.ylabel("Cost")

plt.tight_layout()
plt.draw()
plt.show()

# Graph 2
sns.set_style("darkgrid")
plt.figure(figsize=(18, 10))

plt.plot(np.arange(len(learning_rate) * len(epochs)),
         list(test_r2_sc.values()), color="blue")
plt.xticks(np.arange(len(learning_rate) * len(epochs)),
           labels=test_r2_sc.keys(), rotation='vertical')

plt.title("Testing dataset Learning Rate - Epoch Vs R2_Score")

plt.xlabel("Learning Rate - Epoch")
plt.ylabel("R2_Score")

plt.tight_layout()
plt.draw()
plt.show()

# Graph 3
sns.set_style("darkgrid")
plt.figure(figsize=(18, 10))

plt.plot(np.arange(len(learning_rate) * len(epochs)),
         list(test_mse_sc.values()), color="blue")
plt.xticks(np.arange(len(learning_rate) * len(epochs)),
           labels=test_mse_sc.keys(), rotation=90)

plt.title("Testing dataset Learning Rate - Epoch Vs MSE")

plt.xlabel("Learning Rate - Epoch")
plt.ylabel("MSE")

plt.tight_layout()
plt.draw()
plt.show()

# Calculating Best value Cost, R2_Score and MSE for testing dataset
print("----------Testing dataset----------")

print("Minimum Cost Value on Testing Data: ", min(test_cost_values.values()))
print("Maximum R2-Score Value on Testing Data: ", max(test_r2_sc.values()))
print("Minimum MSE Value on Testing Data: ", min(test_mse_sc.values()))
