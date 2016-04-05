# COMPM041
# Student Name: Zunran Guo
# Student Number: 15118320
# All work is original.

import random
import numpy as np
from scipy import interp
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# define global variables
dataPoint = []  # array of data point
dataPoints = np.zeros(4601)  # numpy array of total data points
dataPointsNormalized = np.zeros(4601)  # numpy array of processed data points
mu = []  # array of mean for each attribute (58 attributes in total)
sd = []  # array of standard deviations for each attribute (58 attributes in total)

# define a method to read the data from file
def loadData(filename):
    print "Loading data..." 
    # randomize the data
    with open(filename) as source:
        data = [ (random.random(), line) for line in source ]
    data.sort()
    
    i = 0      
    for lineInData in data:
        # define local variables
        line = lineInData[1].strip()
        dataPointLocal = []
    
        # store attributes to each data point array
        elements = line.split(',')
        for element in elements:
            dataPointLocal.append(float(element))
        
        # store each data point to the set of total data points
        if(i == 0):
            dataPoint = dataPointLocal
        if(i == 1):
            dataPoints = np.vstack((dataPoint, dataPointLocal))
        if(i > 1):
            dataPoints = np.vstack((dataPoints, dataPointLocal))
         
        # update i
        i += 1
    print("The total number of data points read is %d\n" % (len(dataPoints)))
    return dataPoints

# define a method to segment the data to 10 groups
# 9 groups are combined to be the training set
# 1 group is remained as the test set

# i.e.
# Group 1 will consist of points {1,11,21,...}
# Group 2 will consist of points {2,12,22,...}
# ...
# Group 10 will consist of points {10,20,30,...}
def segmentData(groupNumber, dataPoints):
    dataPointsTraining = np.zeros_like(dataPoints[0])
    
    for n in range (1 , 11):
        if (n == groupNumber):
            dataPointsTesting = np.zeros_like(dataPoints[0])
            j = 0
            while ((groupNumber + j) <= 4601):
                dataPointsTesting = np.vstack((dataPointsTesting, dataPoints[groupNumber - 1 + j]))
                j += 10
        
        if (n != groupNumber):
            i = 0
            while ((n + i) <= 4601):
                dataPointsTraining = np.vstack((dataPointsTraining, dataPoints[n - 1 + i]))
                i += 10
    return dataPointsTesting[1: , :], dataPointsTraining[1: , :]

# define a method to normalize the data
def preconditionData(dataPoints):    
    print "Normalizing all data points..."
    
    N = dataPoints.shape[0] # number of data points
    D = dataPoints.shape[1] # number of dimension of each data point
    
    # calculate means and stand deviations for each attribute
    for i in range(0, D - 1):
        attribute = []
        for j in range (0, N):
            attribute.append(dataPoints[j][i])
        mu.append(np.mean(attribute))
        sd.append(np.std(attribute))
    
    # normalize every attribute
    for i in range(0, D - 1):
        for j in range (0, N):
            dataPoints[j][i] = (dataPoints[j][i] - mu[i]) / sd[i]
            
    # print attribute statistics
    i = 0 
    print "Attribute Statistics:\nAttribute    Mean    Standard Deviation"
    while(i < D - 1):
        print '%d        %f        %f' % (i + 1, mu[i], sd[i])
        i += 1
    return dataPoints
    
# define global constant
ITERATIONS = 500

# hypothesis = w_1x_1 + w_2x_2 + ... w_nx_n + b, where b is a constant
def generate_regression_variables(dataPointsNormalized):
    D = dataPointsNormalized.shape[1] - 1 # number of attributes of each data point
    
    # generate y from dataPointsNormalized
    y = []
    for element in dataPointsNormalized[: , 57:58]:
        y.append(int(element))
    
    # generate X from dataPointsNormalized
    dataExtracted = dataPointsNormalized[: , :57]
    X = np.concatenate((dataExtracted, np.ones((dataExtracted.shape[0], 1))), axis=1)
    
    # initialize all 57 attributes to be zero
    weights = np.zeros((D + 1)) 
    return weights, X, y 
    
def stochastic_gradient_descent(alpha, dataPointsNormalized):
    # initialize variables
    weights, X, y = generate_regression_variables(dataPointsNormalized)
    N = X.shape[0] # number of data points
    msePrevious = 0
    MSEs = []
    
    print("\nPerforming linear regression with stochastic gradient descent \nat a %f learning rate with %d data points..." % (alpha, N))
    for loop in range(0, ITERATIONS):
        for i in range(0, N):     
            hypothesis = np.dot(X, weights)    
            weights = weights + alpha * (y - hypothesis)[i] * X[i]
        
        # calculate the mean squared error (mse)      
        loss = np.dot(X, weights) - y
        mse = np.sum(np.square(loss)) / N
        MSEs.append(mse)
        print("Iteration %d | Cost: %f" % (loop, mse))   
        
        # check whether mse is converged
        if abs(mse - msePrevious) < 0.0001:
            break
        
        # update mse
        msePrevious = mse
    return weights, MSEs

def logistic_stochastic_gradient_descent(alpha, dataPointsNormalized):
    # initialize variables
    weights, X, y = generate_regression_variables(dataPointsNormalized)
    N = X.shape[0] # number of data points
    msePrevious = 0
    MSEs = []
    
    print("\nPerforming logistic regression with stochastic gradient descent \nat a %.8f learning rate with %d data points..." % (alpha, N))
    for loop in range(0, ITERATIONS):
        for i in range(0, N):     
            hypothesis = 1 / (1 + np.exp((-1) * np.dot(X, weights)))    
            weights = weights + alpha * (y - hypothesis)[i] * X[i]
        
        # calculate the mean squared error (mse)      
        loss = np.dot(X, weights) - y
        mse = np.sum(np.square(loss)) / N
        MSEs.append(mse)
        print("Iteration %d | Cost: %f" % (loop, mse))   
        
        # check whether mse is converged
        if abs(mse - msePrevious) < 0.0001:
            break
        
        # update mse
        msePrevious = mse
    return weights, MSEs

def batch_gradient_descent(alpha, dataPointsNormalized):
    # initialize variables
    weights, X, y = generate_regression_variables(dataPointsNormalized)
    N = X.shape[0] # number of data points
    msePrevious = 0
    MSEs = []
    
    print("\nPerforming linear regression with batch gradient descent \nat a %f learning rate with %d data points..." % (alpha, N))
    for loop in range(0, ITERATIONS):
        hypothesis = np.dot(X, weights) 
        summation = 0
        for i in range(0, N):            
            summation += (y - hypothesis)[i] * X[i]
        weights = weights + alpha * summation
        
        # calculate the mean squared error (mse)      
        loss = np.dot(X, weights) - y
        mse = np.sum(np.square(loss)) / N
        MSEs.append(mse)
        print("Iteration %d | Cost: %f" % (loop, mse))   
        
        # check whether mse is converged
        if abs(mse - msePrevious) < 0.001:
            break
        
        # update mse
        msePrevious = mse
    return weights, MSEs

def logistic_batch_gradient_descent(alpha, dataPointsNormalized):
    # initialize variables
    weights, X, y = generate_regression_variables(dataPointsNormalized)
    N = X.shape[0] # number of data points
    msePrevious = 0
    MSEs = []
    
    print("\nPerforming logistic regression with bath gradient descent \nat a %.8f learning rate with %d data points..." % (alpha, N))
    for loop in range(0, ITERATIONS):
        hypothesis = 1 / (1 + np.exp((-1) * np.dot(X, weights))) 
        summation = 0
        for i in range(0, N):            
            summation += (y - hypothesis)[i] * X[i]
        weights = weights + alpha * summation
        
        # calculate the mean squared error (mse)      
        loss = np.dot(X, weights) - y
        mse = np.sum(np.square(loss)) / N
        MSEs.append(mse)
        print("Iteration %d | Cost: %f" % (loop, mse))   
        
        # check whether mse is converged
        if abs(mse - msePrevious) < 0.001:
            break
        
        # update mse
        msePrevious = mse
    return weights, MSEs

def generate_ROC_variables(dataPointsTesting, weights):
    # generate X from dataPointsTesting
    dataExtracted = dataPointsTesting[: , :57]
    X_train = np.concatenate((dataExtracted, np.ones((dataExtracted.shape[0], 1))), axis=1)
    
    # generate y_score and y_test
    y_score = np.dot(X_train, weights)
    y_test = []
    for element in dataPointsTesting[: , 57:58]:
        y_test.append(int(element))
            
    # generate ROC variables
    false_positive_rates, true_positive_rates, _ = roc_curve(y_test, y_score)
    return false_positive_rates, true_positive_rates
    
if __name__ == '__main__':
    # load data
    dataPoints = loadData("spambase.data")
    
    # precondition data
    dataPointsNormalized = preconditionData(dataPoints)
    
    # initialize variables
    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)
      
    # segment normalized data to 10 groups
    # 9 groups are combined to be the training set
    # 1 group is remained as the test set
    for i in range(1, 11):
        print("\nFold %d" % (i))
        dataPointsTesting, dataPointsTraining = segmentData(i, dataPointsNormalized)
      
        # perform linear regression with stochastic gradient descent
        weightsSGD1, MSEsSGD1 = stochastic_gradient_descent(0.001, dataPointsTraining)
#         weightsSGD2, MSEsSGD2 = stochastic_gradient_descent(0.0001, dataPointsTraining)
#         weightsSGD3, MSEsSGD3 = stochastic_gradient_descent(0.00001, dataPointsTraining)
           
        # perform logistic regression with stochastic gradient descent
#         weightsLSGD1, MSEsLSGD1 = logistic_stochastic_gradient_descent(0.000001, dataPointsTraining)
#         weightsLSGD2, MSEsLSGD2 = logistic_stochastic_gradient_descent(0.0000001, dataPointsTraining)
#         weightsLSGD3, MSEsLSGD3 = logistic_stochastic_gradient_descent(0.00000001, dataPointsTraining)
           
        # perform linear regression with batch gradient descent
#         weightsBGD1, MSEsBGD1 = batch_gradient_descent(0.00001, dataPointsTraining)
#         weightsBGD2, MSEsBGD2 = batch_gradient_descent(0.000001, dataPointsTraining)
#         weightsBGD3, MSEsBGD3 = batch_gradient_descent(0.0000001, dataPointsTraining)
       
        # perform logistic regression with batch gradient descent
#         weightsLBGD1, MSEsLBGD1 = logistic_batch_gradient_descent(0.000001, dataPointsTraining)
#         weightsLBGD2, MSEsLBGD2 = logistic_batch_gradient_descent(0.0000001, dataPointsTraining)
#         weightsLBGD3, MSEsLBGD3 = logistic_batch_gradient_descent(0.00000001, dataPointsTraining)
           
        # generate ROC variables
        fpr, tpr = generate_ROC_variables(dataPointsTesting, weightsSGD1)
#         fpr, tpr = generate_ROC_variables(dataPointsTesting, weightsBGD1)
#         fpr, tpr = generate_ROC_variables(dataPointsTesting, weightsLSGD1)
#         fpr, tpr = generate_ROC_variables(dataPointsTesting, weightsLBGD1)
          
        # estimate mean_tpr by interpolation over the range of mean_fpr and sum over all folds
        mean_tpr += interp(mean_fpr, fpr, tpr)
        mean_tpr[0] = 0.0
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, linewidth=1, label='ROC fold %d (area = %0.2f)' % (i, roc_auc))
      
    mean_tpr /= 10
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
      
    # plot Receiver Operating Characteristic (ROC) Curves
    plt.title("Receiver Operating Characteristic (ROC) Curve")
    plt.xlabel("False Positive Rates")
    plt.ylabel("True Positive Rates")
    plt.plot(mean_fpr, mean_tpr, 'k--', label='Mean ROC (area = %0.2f)' % mean_auc, linewidth=4)
    plt.legend(loc='lower right')
    plt.show()
    
    # plot SGD learning curves
    # perform linear regression with stochastic gradient descent
    print("\nNow plotting different stochastic gradient descent learning curves...")
    weightsSGD1, MSEsSGD1 = stochastic_gradient_descent(0.001, dataPointsNormalized)
    weightsSGD2, MSEsSGD2 = stochastic_gradient_descent(0.0001, dataPointsNormalized)
    weightsSGD3, MSEsSGD3 = stochastic_gradient_descent(0.00001, dataPointsNormalized)
         
    # perform logistic regression with stochastic gradient descent
#     weightsLSGD1, MSEsLSGD1 = logistic_stochastic_gradient_descent(0.000001, dataPointsNormalized)
#     weightsLSGD2, MSEsLSGD2 = logistic_stochastic_gradient_descent(0.0000001, dataPointsNormalized)
#     weightsLSGD3, MSEsLSGD3 = logistic_stochastic_gradient_descent(0.00000001, dataPointsNormalized)

    plotSGD = plt.subplot()
    plotSGD.set_title("Stochastic Gradient Descent Learning Curve")
    plotSGD.set_xlabel("Iterations")
    plotSGD.set_ylabel("Mean Squared Error (MSE)")
    plotSGD.plot(np.arange(0, len(MSEsSGD1)), MSEsSGD1, 'o', color='r', label='Learning Rate 0.001')
    plotSGD.plot(np.arange(0, len(MSEsSGD2)), MSEsSGD2, 'o', color='g', label='Learning Rate 0.0001')
    plotSGD.plot(np.arange(0, len(MSEsSGD3)), MSEsSGD3, 'o', color='b', label='Learning Rate 0.00001')
#     plotSGD.plot(np.arange(0, len(MSEsLSGD1)), MSEsLSGD1, 'o', color='r', label='Learning Rate 0.00001')
#     plotSGD.plot(np.arange(0, len(MSEsLSGD2)), MSEsLSGD2, 'o', color='g', label='Learning Rate 0.000001')
#     plotSGD.plot(np.arange(0, len(MSEsLSGD3)), MSEsLSGD3, 'o', color='b', label='Learning Rate 0.0000001')
     
    # plot BGD learning curves
    # perform linear regression with batch gradient descent
#     print("\nNow plotting different batch gradient descent learning curves...")
#     weightsBGD1, MSEsBGD1 = batch_gradient_descent(0.00001, dataPointsNormalized)
#     weightsBGD2, MSEsBGD2 = batch_gradient_descent(0.000001, dataPointsNormalized)
#     weightsBGD3, MSEsBGD3 = batch_gradient_descent(0.0000001, dataPointsNormalized)
     
    # perform logistic regression with batch gradient descent
#     weightsLBGD1, MSEsLBGD1 = logistic_batch_gradient_descent(0.000001, dataPointsNormalized)
#     weightsLBGD2, MSEsLBGD2 = logistic_batch_gradient_descent(0.0000001, dataPointsNormalized)
#     weightsLBGD3, MSEsLBGD3 = logistic_batch_gradient_descent(0.00000001, dataPointsNormalized)
    
#     plotBGD = plt.subplot()
#     plotBGD.set_title("Batch Gradient Descent Learning Curve")
#     plotBGD.set_xlabel("Iterations")
#     plotBGD.set_ylabel("Mean Squared Error (MSE)")
#     plotBGD.plot(np.arange(0, len(MSEsBGD1)), MSEsBGD1, 'o', color='r', label='Learning Rate 0.00001')
#     plotBGD.plot(np.arange(0, len(MSEsBGD2)), MSEsBGD2, 'o', color='g', label='Learning Rate 0.000001')
#     plotBGD.plot(np.arange(0, len(MSEsBGD3)), MSEsBGD3, 'o', color='b', label='Learning Rate 0.0000001')
#     plotBGD.plot(np.arange(0, len(MSEsLBGD1)), MSEsLBGD1, 'o', color='r', label='Learning Rate 0.000001')
#     plotBGD.plot(np.arange(0, len(MSEsLBGD2)), MSEsLBGD2, 'o', color='g', label='Learning Rate 0.0000001')
#     plotBGD.plot(np.arange(0, len(MSEsLBGD3)), MSEsLBGD3, 'o', color='b', label='Learning Rate 0.00000001')
    plt.legend()
    plt.show()
    
    
        
