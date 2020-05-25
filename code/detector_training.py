import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier as KNN
import glob
import sys
import math

VERBOSE = True

def main(train_dir_name, label_file_path, adv_dir_name):
    #input handling
    f_x_train = [np.load(file_name) for file_name in glob.glob(train_dir_name)]
    x_labels = np.load(label_file_path)

    # Compute Euclidian activation spaces for each layer
    pca_array = []
    for i in range(len(f_x_train[0])):
        pca = PCA(n_components=100)
        pca.fit([x[i]for x in f_x_train])
        pca_array.append(pca)

    train_activation_spaces = []
    for X in f_x_train:
        _X = [pca_array[i].transform(X[i]) for i in range(len(f_x_train[0]))]
        #if VERBOSE:
        #    print(pca.explained_variance_ratio_)
        #    print(pca.singular_values_)
        train_activation_spaces.append(_X)

    # Train a k-NN classifier for each activation space
    KNN_classefiers = []
    for i in range(len(train_activation_spaces[0])):
        neigh = KNN(n_neighbors=5)
        neigh.fit([x[i] for x in train_activation_spaces], x_labels)
        KNN_classefiers.append(neigh)

    # Assign input samples with a sequence of class labels
    Y_train_predict = []
    for x in train_activation_spaces:
        Y_train_predict.append([KNN_classefiers[i].predict(x[i]) for i in range(len(x))])

    # Calculate the a priori probability for a classification
    # change at the i position of a sequence
    count_arr = [0] * (len(Y_train_predict[0]) - 1)
    for i in range(1, len(Y_train_predict)):
        for j in range(len(Y_train_predict[i])):
            if Y_train_predict[i][j] != Y_train_predict[i-1][j]:
                count_arr[j] += 1

    count_arr = [float(i)/(len(Y_train_predict)-1) for i in count_arr]

    # Construct adversarial examples (input files)
    x_adv = [np.load(file_name) for file_name in glob.glob(adv_dir_name)]

    advertise_activation_spaces = []
    for X in x_adv:
        _X = pca.transform(X)
        if VERBOSE:
            print(pca.explained_variance_ratio_)
            print(pca.singular_values_)
        advertise_activation_spaces.append(_X)

    Y_advertise_predict = []
    for x in train_activation_spaces:
        Y_advertise_predict.append([KNN_classefiers[i].predict(x[i]) for i in range(len(x))])

    # Calculate class switching Bayesian log likelihood
    Y_arr = Y_train_predict + Y_advertise_predict
    ll_x_arr = [0] * len(Y_arr)
    for i in range(len(Y_arr)):
        for j in range(1, len(Y_arr[i])):
            if Y_arr[i] != Y_arr[i-1]:
                ll_x_arr[i] += math.log(count_arr[i-1])
            else:
                ll_x_arr[i] += math.log(1 - count_arr[i - 1])

    # Calculate the cutoff log likelihood value. Choose an
    # appropriate threshold value by using a ROC curve



if __name__ == '__main__':
    dir_name = sys.argv[1]
    main(dir_name)


