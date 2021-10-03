# Linear Regression using Gradient Descent

## Description:

The purpose of this code is to provide an insight into how the gradient descent algorithm can be used for linear regression by finding the minimum of the total squared error function. Here we want to predict the cost of a house per unit area and we have access to following features:
1 X1 transaction date  
 2 X2 house age  
 3 X3 distance to the nearest MRT station  
 4 X4 number of convenience stores  
 5 X5 latitude  
 6 X6 longitude  
The dataset used here is from UCI and can be accessed directly using the link provided in the code. For this kind of prediction, we need to use Multivariate Linear Regression.

## Requirements & Libraries used:

- Python 3
- Pandas
- Scikit Learn
- NumPy
- Matplotlib
- Seaborn

## Algorithm:

- Import the libraries and load the dataset using the link from UCI
- Remove the 'No' column and checking for any null values available
- Convert feature 'Transaction date' from type float to type int
- Calculate the Correlation and plotting its a graph
- Split X = attributes and Y = Output
- Split Train and Test data with a test size of 0.20
- Normalize the training and testing data
- Defined partial derivate and gradient descent functions
- Call Gradient Descent function for multiple epochs and learning rates on Training and Testing data
- Store the results for each iteration

## Running the code:

- Click on the following links for part1 and part2 respectively to open the Colab notebook
- Part 1 - https://colab.research.google.com/drive/19OV1P0AmI-rRR1rb12lbq5QZiHNbKZqT?usp=sharing
- Part 2 - https://colab.research.google.com/drive/1zXPfqj6QCBYl4XU1BF7bq9CnINOvKlIR?usp=sharing

- To execute the code in each cell, select it with a click and then either press the play button to the left of the code, or use the keyboard shortcut "Command/Ctrl+Enter"

## Report:

- Report contains Best result for both part-1 and part-2.
- Answer to the question - Answer this question: Are you satisfied that you have found the best solution? Explain.
- References

## Log.txt:

- Log file contains the output for different epochs and learning rates.
