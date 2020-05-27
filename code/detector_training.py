import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import glob
import sys
import math
import os

VERBOSE = True
EXT = '.npy'

def main(data_dir_name):
    #input handling
    tmp_1 = 'y_adv_target'
    tmp_2 = 'y_adv_original'

    f_x_train = np.load(os.path.join(data_dir_name, 'logits_on_x_benign' + EXT))
    x_labels  = np.load(os.path.join(data_dir_name, 'y_benign' + EXT))

    # Compute Euclidian activation spaces for each layer
    pca_array = []
    for i in range(len(f_x_train[0])):
        pca = PCA(n_components=20)
        pca.fit([x[i].flatten() for x in f_x_train])
        pca_array.append(pca)

    train_activation_spaces = []
    for X in f_x_train:
        _X = [pca_array[i].transform([X[i].flatten()])[0] for i in range(len(f_x_train[0]))]
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
        Y_train_predict.append([KNN_classefiers[i].predict([x[i]])[0] for i in range(len(x))])

    # Calculate the a priori probability for a classification
    # change at the i position of a sequence
    sum_count_arr = [0] * (len(Y_train_predict[0]) - 1)
    for i in range(len(Y_train_predict)):
        count_arr = [0] * (len(Y_train_predict[0]) - 1)
        for j in range(1, len(Y_train_predict[i])):
            if Y_train_predict[i][j] != Y_train_predict[i][j-1]:
                count_arr[j-1] += 1
        sum_count_arr = [(sum_count_arr[k]+count_arr[k]) for k in range(len(sum_count_arr))]

    p_arr = [float(i)/(len(Y_train_predict)-1) for i in sum_count_arr]

    # Construct adversarial examples (input files)
    x_adv = np.load(os.path.join(data_dir_name, 'logits_on_x_adv' + EXT))

    advertise_activation_spaces = []
    for X in x_adv:
        _X = [pca_array[i].transform([X[i].flatten()])[0] for i in range(len(x_adv[0]))]
        # if VERBOSE:
        #    print(pca.explained_variance_ratio_)
        #    print(pca.singular_values_)
        advertise_activation_spaces.append(_X)

    Y_advertise_predict = []
    for x in advertise_activation_spaces:
        Y_advertise_predict.append([KNN_classefiers[i].predict([x[i]])[0] for i in range(len(x))])

    # Calculate class switching Bayesian log likelihood
    Y_true = [0]*len(Y_train_predict) + [1]*len(Y_advertise_predict)
    Y_arr = Y_train_predict + Y_advertise_predict
    ll_x_arr = [0] * len(Y_arr)
    for i in range(len(Y_arr)):
        for j in range(1, len(Y_arr[i])):
            if Y_arr[i][j] != Y_arr[i][j-1]:
                ll_x_arr[i] += math.log(p_arr[j-1])
            else:
                ll_x_arr[i] += math.log(1 - p_arr[j-1])

    # Calculate the cutoff log likelihood value. Choose an
    # appropriate threshold value by using a ROC curve
    fpr, tpr, thresholds = roc_curve(Y_true, ll_x_arr, pos_label=1)
    roc_auc = auc(fpr, tpr)
    print(fpr)
    print(tpr)
    print(thresholds)
    print(roc_auc)

    #plot ROC
    lw = 2
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.3f)' % roc_auc)
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.02])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic graph')
    plt.legend(loc="lower right")
    plt.show()


if __name__ == '__main__':
    dir_name = sys.argv[1]
    main(dir_name)


