import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import sys
import os


def main(data_dir_name, roc_graph_file_name, precision_recall_graph_file_name, histogram_file_name):
    Y_score = np.load(os.path.join(data_dir_name, 'prediction.npy'))
    Y_score_benign = Y_score[:len(Y_score)//2]
    Y_score_adv = Y_score[len(Y_score)//2:]

    #Y_score_adv = np.load(os.path.join(data_dir_name, 'test_adv_tanh.npy'))
    #Y_score_adv = np.absolute(Y_score_adv)
    #Y_score_benign = np.load(os.path.join(data_dir_name, 'test_benign_tanh.npy'))
    #Y_score_benign = np.absolute(Y_score_benign)
    #Y_score = Y_score_benign.tolist() + Y_score_adv.tolist()
    Y_true = ([0] * len(Y_score_benign)) + ([1] * len(Y_score_adv))

    lw = 2

    lr_precision, lr_recall, _ = precision_recall_curve(Y_true, Y_score, pos_label=1)
    precision_recall_auc = auc(lr_recall, lr_precision)

    plt.plot([0, 1], [1, 0], color='navy', lw=lw, linestyle='--')
    plt.plot(lr_recall, lr_precision, color='darkorange', lw=lw, label='precision recall curve (area = %0.3f)' % precision_recall_auc)
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.02])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision Recall graph')
    plt.legend(loc="lower right")

    plt.savefig(os.path.join(data_dir_name, precision_recall_graph_file_name))
    #plt.show(block=True)
    plt.clf()

    fpr, tpr, thresholds = roc_curve(Y_true, Y_score, pos_label=1)
    roc_auc = auc(fpr, tpr)

    lw = 2
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.3f)' % roc_auc)
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.02])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic graph')
    plt.legend(loc="lower right")

    plt.savefig(os.path.join(data_dir_name, roc_graph_file_name))
    #plt.show(block=True)
    plt.clf()

    #plt.figure(figsize=(50, 50))
    plt.xlabel('test sample #')
    plt.ylabel('likelihood value')
    plt.title('test samples likelihood value')
    plt.axvline(x=len(Y_score)//2, linewidth=lw, color='r', linestyle='--')
    plt.bar(np.arange(len(Y_score_benign)), Y_score_benign, color='navy', label='benign', align='edge')
    plt.bar(np.arange(len(Y_score_adv)) + len(Y_score_benign), Y_score_adv, color='darkorange', label='adversarial',
            align='edge')
    plt.legend()
    plt.savefig(os.path.join(data_dir_name, histogram_file_name))
    # plt.show(block=True)
    plt.clf()

if __name__ == '__main__':
    main('./', 'roc_graph_test.png', 'precision_recall_graph_test.png', 'prediction_score_histogram.png')