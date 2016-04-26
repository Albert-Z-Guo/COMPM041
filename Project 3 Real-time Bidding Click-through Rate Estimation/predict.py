# COMPM041
# Student Name: Zunran Guo
# Student Number: 15118320
# All work is original.

import time
import preprocess
import numpy as np
    
def test_internal(X_train, X_test, y_train, y_test):
    start = time.time()  # start timing
    print '\ntraining... '
    
    # training
    # reference: http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html#sklearn.linear_model.LogisticRegression
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression(C=0.1, fit_intercept=True, max_iter=600, n_jobs=-1, tol=0.000001)
    
    # alternative model
    # reference: http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html#sklearn.linear_model.SGDClassifier
#     from sklearn.linear_model import SGDClassifier
#     model = SGDClassifier(loss='log', n_iter=100, epsilon=0.0001, n_jobs=-1)
    
    model.fit(X_train, y_train)
    print model
    
    # check model accuracy
    from sklearn import metrics
    prediction = model.predict_proba(X_test)[:, 1] # select the column of probabilities of 1 (click response)
    fpr, tpr, thresholds = metrics.roc_curve(y_test.values, prediction, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    print 'AUC score:', auc
    
    end = time.time()  # end timing
    print("total training time: %d seconds" % (end - start))
    
def test(X, y):
    start = time.time()  # start timing
    print '\ntraining... '
    
    # training
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression(C=0.1, fit_intercept=True, max_iter=600, n_jobs=-1, tol=0.000001)
     
    # alternative model
#     from sklearn.linear_model import SGDClassifier
#     model = SGDClassifier(loss='log', n_iter=100, epsilon=0.0001, n_jobs=-1)
    
    model.fit(X, y)
    print model
    
    # save prediction
    X_test = preprocess.precondition('shuffle_data_test.txt', 0)
    prediction = model.predict_proba(X_test)[:, 1] # select the column of probabilities of 1 (click response)
    np.savetxt("prediction_complete_best.csv", np.dstack((np.arange(1, prediction.size + 1), prediction))[0], "%d,%f", header="Id,Prediction")
    
    end = time.time()  # end timing
    print("\ntotal training time: %d seconds" % (end - start))
    
    
if __name__ == '__main__':
    # preprocess and split data
    X, y = preprocess.precondition('data_train.txt', 1)
    X_train, X_test, y_train, y_test = preprocess.sample(X, y)

    # user may choose to load previously preprocessed data
#     print 'loading preprocessed data...'
#     X_train = np.genfromtxt('X_train.csv', delimiter=',')
#     y_train = np.genfromtxt('y_train.csv', delimiter=',')
#     X_test = np.genfromtxt('X_test.csv', delimiter=',')
#     y_test = np.genfromtxt('y_test.csv', delimiter=',')
    
    """
    test_internal(X_train, X_test, y_train, y_test) is an internal testing method 
    that generates an AUC score that measures the prediction accuracy of the model.
    The prediction model is trained using 70% of the labeled data 
    from the complete training data. This prediction model is then tested 
    on the other 30% of the labeled data form the complete training data.  
    
    test(X, y) is the predicting method that trains the prediction model 
    using the complete training data and outputs the final prediction.
    """
    test_internal(X_train, X_test, y_train, y_test) 
#     test(X, y)
    
    
