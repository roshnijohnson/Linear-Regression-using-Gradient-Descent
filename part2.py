# Imported libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# reading dataset from UCI


df = pd.read_excel(
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00477/Real%20estate%20valuation%20data%20set.xlsx")
df

# Checking dataset

print(df.info())

#  Removing the 'No' column

df = df.drop('No', 1)

# checking for null values

df.isna().sum()

# Converting feature 'Transaction date' to type int

df['X1 transaction date'] = df['X1 transaction date'].astype(int)
print(df.head())

# Calculating the Correlation

df.corr()["Y house price of unit area"][:-1:].sort_values(ascending=False)

# Plotting correlation graph

plt.figure(figsize=(10, 8))
sns.heatmap(data=df.corr(), annot=True, cmap="viridis")
plt.show()

# Splitting X (attributes) and Y (Output)

X = df.iloc[:, :-1]

Y = df.iloc[:, -1:]


# Splitting Train and Test data

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42)

# Normalizing the X_train and X_test

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)

# Variables to store results for multiple iteration results on test and training data

learning_rate = np.arange(0.01, 0.1, 0.01)
epochs = np.arange(4000, 10000, 1000)
train_score = {}
test_score = {}
coef = {}
intercept = {}

train_y_pred_vals = {}
train_r2 = {}
train_mse = {}

test_y_pred_vals = {}
test_r2 = {}
test_mse = {}


for learn_rate in learning_rate:
    for epoch in epochs:

        # defining SGDRegressor
        SGD = SGDRegressor(eta0=learn_rate, learning_rate='constant', max_iter=epoch).fit(
            X_train, Y_train.values.ravel())

        # Storing the coefficients and Intercepts
        coef[(learn_rate, epoch)] = SGD.coef_  # Coefs
        intercept[(learn_rate, epoch)] = SGD.intercept_  # intercepts

        # predicting the output on training data

        y_pred = SGD.predict(X_train)
        train_y_pred_vals[(learn_rate, epoch)] = y_pred

        train_score[(learn_rate, epoch)] = SGD.score(
            X_train, Y_train) * 100  # model score on training data

        train_mse[(learn_rate, epoch)] = mean_squared_error(
            Y_train, y_pred)  # Should be as low as possible
        train_r2[(learn_rate, epoch)] = r2_score(Y_train, y_pred) * \
            100  # Should be as high as possible

        # predicting the output on training data

        y_pred = SGD.predict(X_test)
        test_y_pred_vals[(learn_rate, epoch)] = y_pred

        # model score on training data (generalization score)
        test_score[(learn_rate, epoch)] = SGD.score(X_test, Y_test) * 100

        test_mse[(learn_rate, epoch)] = mean_squared_error(
            Y_test, y_pred)  # Should be as low as possible
        test_r2[(learn_rate, epoch)] = r2_score(Y_test, y_pred) * \
            100  # Should be as high as possible

# Calculating Best value Cost, R2_Score and MSE for training dataset

print("Minimum Score Value on Training Data: ", max(train_score.values()))
print("Maximum R2-Score Value on Training Data: ", max(train_r2.values()))
print("Minimum MSE Value on Training Data: ", min(train_mse.values()))

# Calculating Best value Cost, R2_Score and MSE for testing dataset

print("Minimum Score Value on Testing Data: ", max(test_score.values()))
print("Maximum R2-Score Value on Testing Data: ", max(test_r2.values()))
print("Minimum MSE Value on Testing Data: ", min(test_mse.values()))

# Plotting graphs

#  Graph 1
sns.set_style("darkgrid")

plt.figure(figsize=(18, 10))
plt.plot(np.arange(len(learning_rate) * len(epochs)),
         list(train_score.values()), color="red", label="SGD_train")
plt.plot(np.arange(len(learning_rate) * len(epochs)),
         list(test_score.values()), color="blue", label="SGD_test")
plt.xticks(np.arange(len(learning_rate) * len(epochs)),
           labels=train_score.keys(), rotation='vertical')

plt.title("Learning Rate - Epoch Vs R2_Score")

plt.xlabel("Learning Rate - Epoch")
plt.ylabel("R2_Score")
plt.legend(loc="center left", fontsize=55)
plt.tight_layout()
plt.draw()
plt.show()

#  Graph 2
sns.set_style("darkgrid")

plt.figure(figsize=(18, 10))
plt.plot(np.arange(len(learning_rate) * len(epochs)),
         list(train_mse.values()), color="red", label="SGD_train")
plt.plot(np.arange(len(learning_rate) * len(epochs)),
         list(test_mse.values()), color="blue", label="SGD_test")

plt.xticks(np.arange(len(learning_rate) * len(epochs)),
           labels=train_mse.keys(), rotation='vertical')

plt.title("Learning Rate - Epoch Vs MSE")


plt.xlabel("Learning Rate - Epoch")
plt.ylabel("MSE")
plt.legend(loc="center left", fontsize=55)

plt.tight_layout()
plt.draw()
plt.show()
