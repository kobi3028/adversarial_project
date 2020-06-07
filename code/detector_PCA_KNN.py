from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import glob
import sys
import math
import os
import pickle


ADVERSARIAL = 1
NORMAL = 0
DUMP_FILE_EXT = '.pickle'
STRIVES_FOR_ZERO = float(1e-323)

class Detector:

    DEFAULT_FILE_NAME = 'detector'

    def __init__(self, layers_number, samples_number):
        self.pca_array = []
        self.knn_classefiers = []
        self.p_arr = []
        self.cutoff = 0
        # fit process temp data
        self.__benign_activation_spaces = []
        self.__adversarial_activation_spaces = []
        for i in range(samples_number):
            self.__benign_activation_spaces.append([])
            self.__adversarial_activation_spaces.append([])
            for j in range(layers_number):
                self.__benign_activation_spaces[i].append([])
                self.__adversarial_activation_spaces[i].append([])

        self.pca_array = [None] * layers_number

    def partial_fit_activation_spaces(self, layer, partial_benign_trainset):
        if self.pca_array[layer] is None:
            n_components = min(100, len(partial_benign_trainset[0].flatten()))
            i_pca = IncrementalPCA(n_components=n_components, batch_size=n_components//10)
            self.pca_array[layer] = i_pca
        self.pca_array[layer].partial_fit([x.flatten() for x in partial_benign_trainset])

    def compute_benign(self, layer, batch_start, partial_benign_trainset):
        for i in range(len(partial_benign_trainset)):
            self.__benign_activation_spaces[i+batch_start][layer] = self.pca_array[layer].transform([partial_benign_trainset[i].flatten()])[0]

    def compute_adversarial(self, layer, batch_start, partial_adversarial_trainset):
        for i in range(len(partial_adversarial_trainset)):
            self.__adversarial_activation_spaces[i+batch_start][layer] = self.pca_array[layer].transform([partial_adversarial_trainset[i].flatten()])[0]

    def finish_fit(self, benign_labels, *, plot_roc_graph=False, roc_graph_file_name=''):
        # Train a k-NN classifier for each activation space
        for i in range(len(self.__benign_activation_spaces[0])):
            neigh = KNN(n_neighbors=5)
            neigh.fit([x[i] for x in self.__benign_activation_spaces], benign_labels)
            self.knn_classefiers.append(neigh)

        # Assign input samples with a sequence of class labels
        Y_train_predict = []
        for x in self.__benign_activation_spaces:
            Y_train_predict.append([self.knn_classefiers[i].predict([x[i]])[0] for i in range(len(x))])

        # Calculate the a priori probability for a classification
        # change at the i position of a sequence
        sum_count_arr = [0] * (len(Y_train_predict[0]) - 1)
        for i in range(len(Y_train_predict)):
            count_arr = [0] * (len(Y_train_predict[0]) - 1)
            for j in range(1, len(Y_train_predict[i])):
                if Y_train_predict[i][j] != Y_train_predict[i][j - 1]:
                    count_arr[j - 1] += 1
            sum_count_arr = [(sum_count_arr[k] + count_arr[k]) for k in range(len(sum_count_arr))]

        self.p_arr = [(STRIVES_FOR_ZERO + (float(i)/(len(Y_train_predict) - 1))) for i in sum_count_arr]
        print(self.p_arr)
        # Assign adversarial samples with a sequence of class labels
        Y_adversarial_predict = []
        for x in self.__adversarial_activation_spaces:
            Y_adversarial_predict.append([self.knn_classefiers[i].predict([x[i]])[0] for i in range(len(x))])

        # Calculate class switching Bayesian log likelihood
        Y_true = [0] * len(Y_train_predict) + [1] * len(Y_adversarial_predict)
        Y_arr = Y_train_predict + Y_adversarial_predict
        ll_x_arr = [0] * len(Y_arr)
        for i in range(len(Y_arr)):
            for j in range(1, len(Y_arr[i])):
                if Y_arr[i][j] != Y_arr[i][j - 1]:
                    ll_x_arr[i] += math.log(self.p_arr[j - 1])
                else:
                    ll_x_arr[i] += math.log(1 - self.p_arr[j - 1])

        # Calculate the cutoff log likelihood value. Choose an
        # appropriate threshold value by using a ROC curve
        fpr, tpr, thresholds = roc_curve(Y_true, ll_x_arr, pos_label=0)
        roc_auc = auc(fpr, tpr)
        # print(fpr)
        # print(tpr)
        # print(thresholds)
        # print(roc_auc)

        if plot_roc_graph or len(roc_graph_file_name) != 0:
            # plot ROC
            lw = 2
            plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
            plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.3f)' % roc_auc)
            plt.xlim([-0.02, 1.02])
            plt.ylim([-0.02, 1.02])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic graph')
            plt.legend(loc="lower right")
            if len(roc_graph_file_name) != 0:
                plt.savefig(roc_graph_file_name)
            if plot_roc_graph:
                plt.show()

        '''TODO: need to find a way to calc it from the ROC curve'''
        self.cutoff = 10

        # cleanup
        # del self.__adversarial_activation_spaces
        # del self.__benign_activation_spaces

    def fit(self, benign_trainset, adv_trainset, benign_labels, *, plot_roc_graph=False, roc_graph_file_name=''):
        # Compute Euclidian activation spaces for each layer
        for i in range(len(benign_trainset[0])):
            pca = PCA(n_components=20)
            pca.fit([x[i].flatten() for x in benign_trainset])
            # if VERBOSE:
            #    print(pca.explained_variance_ratio_)
            #    print(pca.singular_values_)
            self.pca_array.append(pca)

        self.benign_activation_spaces = []
        for X in benign_trainset:
            _X = [self.pca_array[i].transform([X[i].flatten()])[0] for i in range(len(benign_trainset[0]))]
            self.benign_activation_spaces.append(_X)

        # Train a k-NN classifier for each activation space
        for i in range(len(self.benign_activation_spaces[0])):
            neigh = KNN(n_neighbors=5)
            neigh.fit([x[i] for x in self.benign_activation_spaces], benign_labels)
            self.knn_classefiers.append(neigh)

        # Assign input samples with a sequence of class labels
        Y_train_predict = []
        for x in self.benign_activation_spaces:
            Y_train_predict.append([self.knn_classefiers[i].predict([x[i]])[0] for i in range(len(x))])

        # Calculate the a priori probability for a classification
        # change at the i position of a sequence
        sum_count_arr = [0] * (len(Y_train_predict[0]) - 1)
        for i in range(len(Y_train_predict)):
            count_arr = [0] * (len(Y_train_predict[0]) - 1)
            for j in range(1, len(Y_train_predict[i])):
                if Y_train_predict[i][j] != Y_train_predict[i][j - 1]:
                    count_arr[j - 1] += 1
            sum_count_arr = [(sum_count_arr[k] + count_arr[k]) for k in range(len(sum_count_arr))]

        self.p_arr = [float(i) / (len(Y_train_predict) - 1) for i in sum_count_arr]

        # Construct adversarial examples (input files)
        self.adversarial_activation_spaces = []
        for X in adv_trainset:
            _X = [self.pca_array[i].transform([X[i].flatten()])[0] for i in range(len(adv_trainset[0]))]
            self.adversarial_activation_spaces.append(_X)

        Y_advertise_predict = []
        for x in self.adversarial_activation_spaces:
            Y_advertise_predict.append([self.knn_classefiers[i].predict([x[i]])[0] for i in range(len(x))])

        # Calculate class switching Bayesian log likelihood
        Y_true = [0] * len(Y_train_predict) + [1] * len(Y_advertise_predict)
        Y_arr = Y_train_predict + Y_advertise_predict
        ll_x_arr = [0] * len(Y_arr)
        for i in range(len(Y_arr)):
            for j in range(1, len(Y_arr[i])):
                if Y_arr[i][j] != Y_arr[i][j - 1]:
                    ll_x_arr[i] += math.log(self.p_arr[j - 1])
                else:
                    ll_x_arr[i] += math.log(1 - self.p_arr[j - 1])

        # Calculate the cutoff log likelihood value. Choose an
        # appropriate threshold value by using a ROC curve
        fpr, tpr, thresholds = roc_curve(Y_true, ll_x_arr, pos_label=1)
        roc_auc = auc(fpr, tpr)
        # print(fpr)
        # print(tpr)
        # print(thresholds)
        # print(roc_auc)
        '''TODO: need to find a way to calc it from the ROC curve'''
        self.cutoff = 10

        if plot_roc_graph or len(roc_graph_file_name) != 0:
            # plot ROC
            lw = 2
            plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
            plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.3f)' % roc_auc)
            plt.xlim([-0.02, 1.02])
            plt.ylim([-0.02, 1.02])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic graph')
            plt.legend(loc="lower right")
            if len(roc_graph_file_name) != 0:
                plt.savefig(roc_graph_file_name)
            if plot_roc_graph:
                plt.show()

    def predict(self, sample):
        # Compute a class label sequence for the evaluated input
        sample_layer_labels = [0] * len(sample)
        for i in range(len(sample)):
            # Project activation value to the i activation space
            activation_space = self.pca_array[i].transform([sample[i].flatten()])[0]
            # Compute the i class label
            sample_layer_labels[i] = self.knn_classefiers[i].predict([activation_space])

        # Calculate the Bayesian log likelihood of the class label sequence
        ll_x = 0
        for i in range(1, len(sample)):
            if sample_layer_labels[i] != sample_layer_labels[i-1]:
                ll_x += math.log(self.p_arr[i - 1])
            else:
                ll_x += math.log(1 - self.p_arr[i - 1])

        return ll_x
        # Compare log likelihood against the cutoff value
        # if ll_x < self.cutoff:
        #    return ADVERSARIAL
        # return NORMAL

    def predict_range(self, sample_arr):
        return [self.predict(sample) for sample in sample_arr]

    def dump(self, data_dir_name, file_name):
        with open(os.path.join(data_dir_name, file_name + DUMP_FILE_EXT), 'wb') as f:
            pickle.dump(self, f)

    def load(self, data_dir_name, file_name):
        with open(os.path.join(data_dir_name, file_name), 'rb') as f:
            d = pickle.load(f)
            self.pca_array = d.pca_array
            self.knn_classefiers = d.knn_classefiers
            self.p_arr = d.p_arr
            self.cutoff = d.cutoff


