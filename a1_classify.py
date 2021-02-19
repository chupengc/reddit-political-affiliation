#  * This code is provided solely for the personal and private use of students
#  * taking the CSC401 course at the University of Toronto. Copying for purposes
#  * other than this use is expressly prohibited. All forms of distribution of
#  * this code, including but not limited to public repositories on GitHub,
#  * GitLab, Bitbucket, or any other online platform, whether as given or with
#  * any changes, are expressly prohibited.
#
#  * All of the files in this directory and all subdirectories are:
#  * Copyright (c) 2020 Frank Rudzicz

import argparse
import os
# import time

from scipy.stats import ttest_rel
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import SelectKBest
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier

# set the random state for reproducibility
import numpy as np

np.random.seed(401)


def accuracy(C):
    """ Compute accuracy given Numpy array confusion matrix C.
    Returns a floating point value
    """
    n = C.shape[0]
    corr = 0
    for i in range(n):
        corr += C[i][i]

    return corr / np.sum(C)


def recall(C):
    """ Compute recall given Numpy array confusion matrix C.
    Returns a list of floating point values
    """
    n = C.shape[0]
    rec = []
    sums = np.sum(C, axis=0)
    for i in range(n):
        corr = C[i][i]
        rec.append(corr / sums[i])

    return rec


def precision(C):
    """ Compute precision given Numpy array confusion matrix C.
    Returns a list of floating point values
    """
    n = C.shape[0]
    prec = []
    sums = np.sum(C, axis=1)
    for i in range(n):
        corr = C[i][i]
        prec.append(corr / sums[i])

    return prec


def train_classifier(X_train, X_test, y_train, y_test):
    """ Train all five classifiers using the training and testing samples

    :return: SGD_C, Gaussian_C, RFC_C, MLP_C, Ada_C: confusion matrix of each
    classifier
    """
    SGD_model = SGDClassifier()
    SGD_model.fit(X_train, y_train)
    SGD_y = SGD_model.predict(X_test)
    SGD_C = confusion_matrix(y_test, SGD_y)

    Gaussian_model = GaussianNB()
    Gaussian_model.fit(X_train, y_train)
    Gaussian_y = Gaussian_model.predict(X_test)
    Gaussian_C = confusion_matrix(y_test, Gaussian_y)

    RFC_model = RandomForestClassifier()
    RFC_model.fit(X_train, y_train)
    RFC_y = RFC_model.predict(X_test)
    RFC_C = confusion_matrix(y_test, RFC_y)

    MLP_model = MLPClassifier(alpha=0.05)
    MLP_model.fit(X_train, y_train)
    MLP_y = MLP_model.predict(X_test)
    MLP_C = confusion_matrix(y_test, MLP_y)

    Ada_model = AdaBoostClassifier()
    Ada_model.fit(X_train, y_train)
    Ada_y = Ada_model.predict(X_test)
    Ada_C = confusion_matrix(y_test, Ada_y)

    return SGD_C, Gaussian_C, RFC_C, MLP_C, Ada_C


def class31(output_dir, X_train, X_test, y_train, y_test):
    """ This function performs experiment 3.1

    Parameters
       output_dir: path of directory to write output to
       X_train: NumPy array, with the selected training features
       X_test: NumPy array, with the selected testing features
       y_train: NumPy array, with the selected training classes
       y_test: NumPy array, with the selected testing classes

    Returns:
       i: int, the index of the supposed best classifier
    """
    SGD_C, Gaussian_C, RFC_C, MLP_C, Ada_C = train_classifier(X_train, X_test,
                                                              y_train, y_test)

    SGD = ["SGDClassifier", accuracy(SGD_C), recall(SGD_C), precision(SGD_C),
           SGD_C]

    Gaussian = ["GaussianNB", accuracy(Gaussian_C), recall(Gaussian_C),
                precision(Gaussian_C), Gaussian_C]

    RFC = ["RandomForestClassifier", accuracy(RFC_C), recall(RFC_C),
           precision(RFC_C), RFC_C]

    MLP = ["MLPClassifier", accuracy(MLP_C), recall(MLP_C), precision(MLP_C),
           MLP_C]

    Ada = ["AdaBoostClassifier", accuracy(Ada_C), recall(Ada_C),
           precision(Ada_C), Ada_C]

    classifiers = [SGD, Gaussian, RFC, MLP, Ada]

    with open(f"{output_dir}/a1_3.1.txt", "w") as outf:
        # For each classifier, compute results and write the following output:
        for i in range(len(classifiers)):
            classifier = classifiers[i]
            outf.write(f'Results for {classifier[0]}:\n')  # Classifier name
            outf.write(f'\tAccuracy: {classifier[1]:.4f}\n')
            outf.write(
                f'\tRecall: {[round(item, 4) for item in classifier[2]]}\n')
            outf.write(
                f'\tPrecision: {[round(item, 4) for item in classifier[3]]}\n')
            outf.write(f'\tConfusion Matrix: \n{classifier[4]}\n\n')
        pass

    acc = [SGD[1], Gaussian[1], RFC[1], MLP[1], Ada[1]]
    iBest_value = max(acc)
    iBest = acc.index(iBest_value)

    return iBest


def class32(output_dir, X_train, X_test, y_train, y_test, iBest):
    """ This function performs experiment 3.2

    Parameters:
       output_dir: path of directory to write output to
       X_train: NumPy array, with the selected training features
       X_test: NumPy array, with the selected testing features
       y_train: NumPy array, with the selected training classes
       y_test: NumPy array, with the selected testing classes
       iBest: int, the index of the supposed best classifier (from task 3.1)

    Returns:
       X_1k: numPy array, just 1K rows of X_train
       y_1k: numPy array, just 1K rows of y_train
   """
    if iBest == 0:
        model = SGDClassifier()
    elif iBest == 1:
        model = GaussianNB()
    elif iBest == 2:
        model = RandomForestClassifier()
    elif iBest == 3:
        model = MLPClassifier(alpha=0.05)
    else:
        model = AdaBoostClassifier()

    rows = X_train.shape[0]
    nums = [1000, 5000, 10000, 15000, 20000]
    test_acc = []
    for n in nums:
        index = np.random.choice(rows, size=n, replace=False)
        train_X = X_train[index, :]
        train_y = y_train[index,]
        model.fit(train_X, train_y)
        model_pred = model.predict(X_test)
        confusion_mat = confusion_matrix(y_test, model_pred)
        acc = accuracy(confusion_mat)
        test_acc.append(acc)

    ind = np.random.choice(rows, size=1000, replace=False)
    X_1k = X_train[ind, :]
    y_1k = y_train[ind,]

    with open(f"{output_dir}/a1_3.2.txt", "w") as outf:
        # For each number of training examples, compute results and write
        # the following output:
        for n in range(len(test_acc)):
            outf.write(f'{nums[n]}: {test_acc[n]:.4f}\n')
        outf.write("Given the data, an increase in the number of "
                   "training samples can lead to an increasing trend in "
                   "accuracy. This could be the result of small number of "
                   "training data not being sufficient to represent the entire "
                   "dataset. Hence, a reasonable increase in the number of "
                   "training samples in a given model is expected to result in "
                   "an increase in accuracy.\n")
        pass

    return (X_1k, y_1k)


def class33(output_dir, X_train, X_test, y_train, y_test, i, X_1k, y_1k):
    """ This function performs experiment 3.3

    Parameters:
       output_dir: path of directory to write output to
       X_train: NumPy array, with the selected training features
       X_test: NumPy array, with the selected testing features
       y_train: NumPy array, with the selected training classes
       y_test: NumPy array, with the selected testing classes
       i: int, the index of the supposed best classifier (from task 3.1)
       X_1k: numPy array, just 1K rows of X_train (from task 3.2)
       y_1k: numPy array, just 1K rows of y_train (from task 3.2)
    """
    # 3.1 find best k features for 32k training set
    k_feats = [5, 50]
    pp_val = {}
    for k in k_feats:
        selector = SelectKBest(f_classif, k=k)
        X_new = selector.fit_transform(X_train, y_train)
        pp = selector.pvalues_
        pp_val[k] = pp

    # 3.2 training the best classifier for 1k and 32k using the best 5 features
    if i == 0:
        model = SGDClassifier()
    elif i == 1:
        model = GaussianNB()
    elif i == 2:
        model = RandomForestClassifier()
    elif i == 3:
        model = MLPClassifier(alpha=0.05)
    else:
        model = AdaBoostClassifier()

    selector = SelectKBest(f_classif, k=5)
    X_1k_new = selector.fit_transform(X_1k, y_1k)
    index_1k = list(selector.get_support(indices=True))
    model.fit(X_1k_new, y_1k)
    model_pred_1k = model.predict(X_test[:, index_1k])
    C_1k = confusion_matrix(y_test, model_pred_1k)
    accuracy_1k = accuracy(C_1k)

    X_new = selector.fit_transform(X_train, y_train)
    index_full = list(selector.get_support(indices=True))
    model.fit(X_new, y_train)
    model_pred = model.predict(X_test[:, index_full])
    C = confusion_matrix(y_test, model_pred)
    accuracy_full = accuracy(C)

    # 3.3 intersection of feature indices from 1k and 32k
    feature_intersection = set(set(index_1k) & set(index_full))

    # 3.4 top 5 features for 32k
    top_5 = index_full

    p_dic = {5: [], 50: []}
    for i in index_1k:
        p_dic[5].append(pp_val[5][i])
    for j in index_full:
        p_dic[50].append(pp_val[50][j])

    with open(f"{output_dir}/a1_3.3.txt", "w") as outf:
        # Prepare the variables with corresponding names, then uncomment
        # this, so it writes them to outf.
        # for each number of features k_feat, write the p-values for
        # that number of features:
        for k_feat in k_feats:
            p_values = p_dic[k_feat]
            outf.write(
                f'{k_feat} p-values: {[format(pval) for pval in p_values]}\n')
        outf.write(f'Accuracy for 1k: {accuracy_1k:.4f}\n')
        outf.write(f'Accuracy for full dataset: {accuracy_full:.4f}\n')
        outf.write(f'Chosen feature intersection: {feature_intersection}\n')
        outf.write(f'Top-5 at higher: {top_5}\n\n')
        pass


def class34(output_dir, X_train, X_test, y_train, y_test, i):
    """ This function performs experiment 3.4

    Parameters
       output_dir: path of directory to write output to
       X_train: NumPy array, with the selected training features
       X_test: NumPy array, with the selected testing features
       y_train: NumPy array, with the selected training classes
       y_test: NumPy array, with the selected testing classes
       i: int, the index of the supposed best classifier (from task 3.1)
    """
    # train each classifier using the k-folds
    kf = KFold(n_splits=5, shuffle=True)
    kf_acc = []
    X = np.concatenate((X_train, X_test), axis=0)
    y = np.concatenate((y_train, y_test), axis=0)
    for train_index, test_index in kf.split(X):
        X_train_kf, X_test_kf = X[train_index], X[test_index]
        y_train_kf, y_test_kf = y[train_index], y[test_index]
        SGD_C, Gaussian_C, RFC_C, MLP_C, Ada_C = \
            train_classifier(X_train_kf, X_test_kf, y_train_kf, y_test_kf)
        acc = [accuracy(SGD_C), accuracy(Gaussian_C), accuracy(RFC_C),
               accuracy(MLP_C), accuracy(Ada_C)]
        kf_acc.append(acc)

    # determine the best classifier
    acc_sum = np.array(kf_acc).sum(axis=0)
    best_index = acc_sum.argmax()

    # calculate the p values
    classifier_acc = []
    for i in range(5):
        classifier_acc.append(np.array(kf_acc)[:, i])

    best = classifier_acc[best_index]
    p_values = []
    for i in range(5):
        if i != best_index:
            S = ttest_rel(best, classifier_acc[i])
            p_values.append(S.pvalue)

    with open(f"{output_dir}/a1_3.4.txt", "w") as outf:
        for fold in kf_acc:
            outf.write(f'Kfold Accuracies: {[round(acc, 4) for acc in fold]}\n')
        outf.write(f'p-values: {[round(pval, 4) for pval in p_values]}\n')
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", help="the input npz file from Task 2",
                        required=True)
    parser.add_argument(
        "-o", "--output_dir",
        help="The directory to write a1_3.X.txt files to.",
        default=os.path.dirname(os.path.dirname(__file__)))
    args = parser.parse_args()

    # TODO: load data and split into train and test.
    data = np.load(args.input)['arr_0']
    X = data[:, 0:-1]
    y = data[:, data.shape[1] - 1]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                        shuffle=True)
    # TODO : complete each classification experiment, in sequence.
    # start = time.time()
    iBest = class31(args.output_dir, X_train, X_test, y_train, y_test)
    X_1k, y_1k = class32(args.output_dir, X_train, X_test, y_train, y_test, iBest)
    class33(args.output_dir, X_train, X_test, y_train, y_test, iBest, X_1k, y_1k)
    class34(args.output_dir, X_train, X_test, y_train, y_test, iBest)
    # end = time.time()
    # print(end - start)
